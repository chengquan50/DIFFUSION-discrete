import torch
import torch.nn as nn
from models.transformer_blocks import GraphColoringTorsoDeepAttn

class GraphColoringFeatureExtractor(nn.Module):
    def __init__(self, num_transformer_layers=1, num_heads=2, d_model=64,
                 dim_feedforward=256, dropout=0.1, max_colors=20):
        super().__init__()
        self.d_model = d_model
        self.max_colors = max_colors
        self.color_embed = nn.Embedding(max_colors+1, d_model//2)
        self.node_linear = nn.Linear(d_model//2 + 1, d_model)
        self.color_linear = nn.Linear(d_model//2, d_model)
        self.deep_attn = GraphColoringTorsoDeepAttn(
            d_model=d_model,
            num_transformer_layers=num_transformer_layers,
            num_heads=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )

    def build_self_attention_mask_nodes(self, adj_matrix):
        B, N, _ = adj_matrix.shape
        I = torch.eye(N, device=adj_matrix.device, dtype=torch.bool).unsqueeze(0)
        AplusI = (adj_matrix > 0) | I
        mask = torch.zeros_like(AplusI, dtype=torch.float)
        mask[~AplusI] = float('-inf')
        return mask

    def build_self_attention_mask_colors(self, N_colors):
        device = self.color_embed.weight.device
        mask = torch.zeros((N_colors, N_colors), dtype=torch.float, device=device)
        return mask.unsqueeze(0)

    def build_cross_mask_node_to_color(self, colors_array):
        B, N = colors_array.shape
        colors_expanded = colors_array.unsqueeze(-1)
        color_indices = torch.arange(N, device=colors_array.device).view(1, 1, N)
        mask_bool = (colors_expanded == color_indices)
        mask = torch.where(mask_bool, torch.tensor(0.0, device=colors_array.device),
                           torch.tensor(float('-inf'), device=colors_array.device))
        row_uncolored = (colors_array == -1).unsqueeze(-1)
        mask = torch.where(row_uncolored, torch.tensor(0.0, device=colors_array.device), mask)
        return mask

    def build_cross_mask_color_to_node(self, colors_array):
        B, N = colors_array.shape
        colors_expanded = colors_array.unsqueeze(1)
        color_indices = torch.arange(N, device=colors_array.device).view(1, N, 1)
        mask_bool = (colors_expanded == color_indices)
        mask = torch.where(mask_bool, torch.tensor(0.0, device=colors_array.device),
                           torch.tensor(float('-inf'), device=colors_array.device))
        col_uncolored = (colors_array == -1).unsqueeze(1)
        mask = torch.where(col_uncolored, torch.tensor(0.0, device=colors_array.device), mask)
        return mask

    def forward(self, observation: dict):
        adj_matrix = observation["adj_matrix"]
        colors = observation.get("colors", None)
        if colors is None:
            B, N, _ = adj_matrix.shape
            colors = -torch.ones(B, N, dtype=torch.long, device=adj_matrix.device)
        B, N = colors.shape
        clamped = torch.clamp(colors+1, min=0, max=self.max_colors)
        color_emb = self.color_embed(clamped)
        is_colored = (colors >= 0).float().unsqueeze(-1)
        node_feats = torch.cat([color_emb, is_colored], dim=-1)
        node_embeddings = self.node_linear(node_feats)
        color_ids = torch.arange(N, device=node_embeddings.device).unsqueeze(0).expand(B, N)
        color_emb = self.color_embed(torch.clamp(color_ids+1, 0, self.max_colors))
        color_embeddings = self.color_linear(color_emb)
        node_self_mask = self.build_self_attention_mask_nodes(adj_matrix)
        color_self_mask = self.build_self_attention_mask_colors(N).repeat(B, 1, 1)
        node2color_mask = self.build_cross_mask_node_to_color(colors)
        color2node_mask = self.build_cross_mask_color_to_node(colors)
        node_emb_out, color_emb_out = self.deep_attn(
            node_embeddings, color_embeddings,
            node_self_mask, color_self_mask,
            node2color_mask, color2node_mask
        )
        if "current_node_index" in observation:
            cni = observation["current_node_index"]
            gather_idx = cni.view(-1, 1, 1).expand(-1, 1, self.d_model)
            current_node_emb = node_emb_out.gather(dim=1, index=gather_idx).squeeze(1)
        else:
            current_node_emb = node_emb_out.mean(dim=1)
        return current_node_emb
