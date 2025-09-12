import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.feature_extractor import GraphColoringFeatureExtractor

def extract(arr, t, x_shape):
    batch_size = t.shape[0]
    out = arr[t].reshape(batch_size, *((1,) * (len(x_shape) - 1)))
    return out

def cosine_beta_schedule(timesteps, s=0.008, dtype=torch.float32):
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x/steps)+s)/(1+s)*np.pi*0.5)**2
    alphas_cumprod = alphas_cumprod/alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:]/alphas_cumprod[:-1])
    betas_clipped = np.clip(betas, a_min=0, a_max=0.999)
    return torch.tensor(betas_clipped, dtype=dtype)


# =============== 6) DiffusionPolicy ===============
class DiffusionPolicy(nn.Module):
    def __init__(self,
                 state_dim, action_dim,
                 noise_ratio=0.1,
                 beta_schedule='linear',
                 n_timesteps=50,
                 clip_denoised=True,
                 predict_epsilon=False,
                 temperature=1.0,
                 device='cpu',
                 loss_type='cross_entropy',
                 use_ddim=False,
                 ddim_steps=10,
                 ddim_eta=0.0,
                 hidden_size=128,
                 attn_output_size=64):
        super(DiffusionPolicy, self).__init__()
        self.use_ddim = use_ddim
        self.ddim_steps = ddim_steps
        self.ddim_eta = ddim_eta
        self.feature_extractor = GraphColoringFeatureExtractor()
        self.state_proj = nn.Linear(attn_output_size, state_dim)
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.n_timesteps = n_timesteps
        self.clip_denoised = clip_denoised
        self.predict_epsilon = predict_epsilon
        self.temperature = temperature
        self.device = device
        self.model = nn.Sequential(
            nn.Linear(state_dim+action_dim+1, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_dim)
        )
        if beta_schedule=='linear':
            betas = linear_beta_schedule_ddpm(n_timesteps)
        elif beta_schedule=='cosine':
            betas = cosine_beta_schedule(n_timesteps)
        elif beta_schedule=='vp':
            betas = vp_beta_schedule(n_timesteps)
        else:
            betas = linear_beta_schedule_ddpm(n_timesteps)
        self.register_buffer('betas', betas)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]], dim=0)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)
        self.register_buffer('posterior_log_variance_clipped', torch.log(torch.clamp(posterior_variance, min=1e-20)))
        self.register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1.-alphas_cumprod))
        self.register_buffer('posterior_mean_coef2', (1.-alphas_cumprod_prev) * torch.sqrt(betas) / (1.-alphas_cumprod))
        self.noise_ratio = noise_ratio
        if loss_type=='mse':
            self.loss_fn = nn.MSELoss()
        elif loss_type=='cross_entropy':
            self.loss_fn = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError("Loss type not implemented.")

    def predict_start_from_noise(self, x_t, t, noise):
        if self.predict_epsilon:
            return (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape)*x_t - 
                    extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)*noise)
        else:
            return noise

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (extract(self.posterior_mean_coef1, t, x_t.shape)*x_start +
                          extract(self.posterior_mean_coef2, t, x_t.shape)*x_t)
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def model_step(self, x, t, s):
        t = t.float().unsqueeze(-1)
        inp = torch.cat([s, x, t], dim=-1)
        return self.model(inp)

    def p_mean_variance(self, x, t, s):
        noise_pred = self.model_step(x, t, s)
        x_recon = self.predict_start_from_noise(x, t, noise_pred)
        if self.clip_denoised:
            x_recon = torch.clamp(x_recon, -10., 10.)
        mean, var, log_var = self.q_posterior(x_recon, x, t)
        return mean, var, log_var

    def p_sample(self, x, t, s):
        model_mean, _, model_log_variance = self.p_mean_variance(x, t, s)
        noise = torch.randn_like(x)
        nonzero_mask = (1 - (t==0).float()).reshape(x.shape[0], *((1,)*(len(x.shape)-1)))
        return model_mean + nonzero_mask * (0.5*model_log_variance).exp()*noise*self.noise_ratio

    def ddim_sample(self, s):
        batch_size = s.size(0)
        shape = (batch_size, self.action_dim)
        x = torch.randn(shape, device=self.device)
        times = torch.linspace(0, self.n_timesteps-1, self.ddim_steps+1).long().tolist()
        time_pairs = list(zip(times[:-1], times[1:]))
        for t_prev, t_cur in reversed(time_pairs):
            t = torch.full((batch_size,), t_prev, dtype=torch.long, device=self.device)
            pred_noise = self.model_step(x, t, s)
            x0 = self.predict_start_from_noise(x, t, pred_noise)
            if self.clip_denoised:
                x0 = torch.clamp(x0, -10., 10.)
            eps = (x - x0*extract(self.sqrt_alphas_cumprod, t, x.shape)) / extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape)
            sigma = self.ddim_eta * torch.sqrt(extract(self.posterior_variance, t, x.shape) *
                    (1 - extract(self.alphas_cumprod_prev, t, x.shape))/(1 - extract(self.alphas_cumprod, t, x.shape)))
            alpha_cumprod_prev = extract(self.alphas_cumprod_prev, t, x.shape)
            x0_coeff = torch.sqrt(alpha_cumprod_prev)
            dir_coeff = torch.sqrt(1 - alpha_cumprod_prev - sigma**2)
            eps_coeff = sigma
            x_prev = x0_coeff * x0 + dir_coeff * eps + eps_coeff * torch.randn_like(x)
            x = x_prev
        return x

    def p_sample_loop(self, s, shape, add_noise=True):
        if self.use_ddim:
            return self.ddim_sample(s)
        else:
            x = torch.randn(shape, device=self.device)
            for i in reversed(range(self.n_timesteps)):
                t = torch.full((shape[0],), i, device=self.device, dtype=torch.long)
                x = self.p_sample(x, t, s)
            return x

    def diffusion_forward(self, projected_state):
        batch_size = projected_state.size(0)
        shape = (batch_size, self.action_dim)
        x = self.p_sample_loop(projected_state, shape, add_noise=True)
        return x

    def encode_states_batch(self, states_list: list):
        adj_matrices = []
        for state in states_list:
            adj = state["adj_matrix"]
            if not torch.is_tensor(adj):
                adj = torch.tensor(adj, device=self.device, dtype=torch.float32)
            else:
                adj = adj.to(self.device)
            adj_matrices.append(adj)
        batched_adj = torch.stack(adj_matrices, dim=0)
        batched_state = {"adj_matrix": batched_adj}
        features = self.feature_extractor(batched_state)
        projected_state = self.state_proj(features)
        return projected_state

    def encode_state(self, state_dict):
        adj = state_dict['adj_matrix']
        if not torch.is_tensor(adj):
            adj = torch.tensor(adj, device=self.device, dtype=torch.float32)
        else:
            adj = adj.to(self.device)
        state_input = {"adj_matrix": adj.unsqueeze(0)}
        features = self.feature_extractor(state_input)
        projected_state = self.state_proj(features)
        return projected_state

    def forward(self, state_dict, mask=None, eval=False):
        projected_state = self.encode_state(state_dict)
        raw_logits = self.diffusion_forward(projected_state)
        final_logits = raw_logits.clone()
        if mask is not None:
            final_logits = final_logits.masked_fill(~mask, -1e10)
        if eval:
            final_probs = F.softmax(final_logits, dim=-1)
        else:
            final_probs = F.softmax(final_logits/self.temperature, dim=-1)
        return final_probs, raw_logits

    @torch.no_grad()
    def sample_action(self, state_dict, mask=None, eval=False):
        final_probs, _ = self.forward(state_dict, mask=mask, eval=eval)
        log_probs = torch.log(final_probs+1e-8)
        if eval:
            action = final_probs.argmax(dim=-1)
        else:
            action = torch.multinomial(final_probs, 1).squeeze(-1)
        selected_log_prob = log_probs.gather(1, action.unsqueeze(-1)).squeeze(-1)
        return action.cpu().numpy(), selected_log_prob.cpu().numpy()

    def train_forward(self, projected_state, mask=None, eval=False):
        raw_logits = self.diffusion_forward(projected_state)
        final_logits = raw_logits.clone()
        if mask is not None:
            final_logits = final_logits.masked_fill(~mask, -1e10)
        if eval:
            final_probs = F.softmax(final_logits, dim=-1)
        else:
            final_probs = F.softmax(final_logits/self.temperature, dim=-1)
        return final_probs, raw_logits

    def p_losses(self, x_start, state, t, weights=1.0):
        x_noisy = self.q_sample(x_start, t)
        t = t.float().unsqueeze(-1)
        pred_x_start = self.model(torch.cat([state, x_noisy, t], dim=-1))
        if self.clip_denoised:
            pred_x_start = torch.clamp(pred_x_start, -10., 10.)
        target_actions = x_start.argmax(dim=-1)
        loss = self.loss_fn(pred_x_start, target_actions)
        return loss

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        sample = (extract(self.sqrt_alphas_cumprod, t, x_start.shape)*x_start +
                  extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)*noise)
        return sample

    def loss(self, x, state, weights=1.0):
        B = x.size(0)
        t = torch.randint(0, self.n_timesteps, (B,), device=x.device).long()
        return self.p_losses(x, state, t, weights=weights)
