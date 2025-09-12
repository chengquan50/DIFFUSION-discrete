import torch
import torch.nn as nn
import torch.nn.functional as F
from models.feature_extractor import GraphColoringFeatureExtractor
def gumbel_softmax_sample(logits, temperature=1.0, hard=False):
    eps = 1e-20
    U = torch.rand_like(logits)
    gumbel_noise = -torch.log(-torch.log(U+eps)+eps)
    y = (logits+gumbel_noise)/temperature
    return F.softmax(y, dim=-1) if not hard else F.gumbel_softmax(logits, tau=temperature, hard=True)


class MLPPolicy(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=128, temperature=1.0, device='cpu'):
        super(MLPPolicy, self).__init__()
        self.device = device
        self.temperature = temperature
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.feature_extractor = GraphColoringFeatureExtractor()
        self.state_proj = nn.Linear(64, state_dim)  # 64 是 feature extractor 输出维度

        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_dim)
        )

    def encode_state(self, state_dict):
        adj = state_dict['adj_matrix']
        if not torch.is_tensor(adj):
            adj = torch.tensor(adj, dtype=torch.float32, device=self.device)
        else:
            adj = adj.to(self.device)
        if adj.ndim == 2:
            adj = adj.unsqueeze(0)  # (1, N, N)
        features = self.feature_extractor({'adj_matrix': adj})  # (1, 64)
        projected_state = self.state_proj(features)  # (1, state_dim)
        return projected_state

    def encode_states_batch(self, states_list):
        adj_matrices = []
        for state in states_list:
            if isinstance(state, dict):
                adj = state["adj_matrix"]
            else:
                raise TypeError(f"Expected dict in state list, got {type(state)}")
            if not torch.is_tensor(adj):
                adj = torch.tensor(adj, dtype=torch.float32, device=self.device)
            else:
                adj = adj.to(self.device)
            adj_matrices.append(adj)
        batched_adj = torch.stack(adj_matrices, dim=0)  # (B, N, N)
        batched_state = {"adj_matrix": batched_adj}
        features = self.feature_extractor(batched_state)  # (B, 64)
        projected_state = self.state_proj(features)  # (B, state_dim)
        return projected_state

    def forward(self, state_dict, mask=None, eval=False):
        state = self.encode_state(state_dict)  # (1, state_dim)
        raw_logits = self.net(state)  # (1, action_dim)
        final_logits = raw_logits.clone()
        if mask is not None:
            final_logits = final_logits.masked_fill(~mask, -1e10)
        if eval:
            final_probs = F.softmax(final_logits, dim=-1)
        else:
            final_probs = gumbel_softmax_sample(final_logits, temperature=self.temperature, hard=False)
        return final_probs, raw_logits

    @torch.no_grad()
    def sample_action(self, state_dict, mask=None, eval=False):
        final_probs, _ = self.forward(state_dict, mask=mask, eval=eval)
        log_probs = torch.log(final_probs + 1e-8)
        if eval:
            action = final_probs.argmax(dim=-1)
        else:
            action = torch.multinomial(final_probs, 1).squeeze(-1)
        selected_log_prob = log_probs.gather(1, action.unsqueeze(-1)).squeeze(-1)
        return action.cpu().numpy(), selected_log_prob.cpu().numpy()

    def train_forward(self, projected_state, mask=None, eval=False):
        raw_logits = self.net(projected_state)  # (B, action_dim)
        final_logits = raw_logits.clone()
        if mask is not None:
            final_logits = final_logits.masked_fill(~mask, -1e10)
        if eval:
            final_probs = F.softmax(final_logits, dim=-1)
        else:
            final_probs = F.softmax(final_logits / self.temperature, dim=-1)
        return final_probs, raw_logits

    def loss(self, x, projected_state, weights=1.0):
        """
        Args:
            x: Tensor (B, action_dim), e.g., one-hot or soft label
            projected_state: Tensor (B, state_dim)
            weights: float or Tensor of shape (B,)
        """
        logits = self.net(projected_state)  # (B, action_dim)
        target_actions = x.argmax(dim=-1)   # (B,)
        loss = F.cross_entropy(logits, target_actions, reduction='none')  # (B,)
        if isinstance(weights, float) or weights.numel() == 1:
            return loss.mean() * weights
        else:
            return (loss * weights).mean()

