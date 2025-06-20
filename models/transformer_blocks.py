import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerBlock_v1(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=256, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, Q, K, V, attn_mask=None):
        if attn_mask is not None and attn_mask.dim() == 3:
            nhead = self.self_attn.num_heads
            attn_mask = attn_mask.repeat(nhead, 1, 1)
        attn_out, _ = self.self_attn(Q, K, V, attn_mask=attn_mask)
        x = self.norm1(Q + self.dropout(attn_out))
        ff = self.linear2(F.relu(self.linear1(x)))
        out = self.norm2(x + self.dropout(ff))
        return out


class GraphColoringTorsoDeepAttn(nn.Module):
    def __init__(self, d_model=64, num_transformer_layers=1, num_heads=2, dim_feedforward=256, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_transformer_layers
        self.node_self_blocks = nn.ModuleList([
            TransformerBlock_v1(d_model, num_heads, dim_feedforward, dropout) for _ in range(num_transformer_layers)
        ])
        self.color_self_blocks = nn.ModuleList([
            TransformerBlock_v1(d_model, num_heads, dim_feedforward, dropout) for _ in range(num_transformer_layers)
        ])
        self.node_color_blocks = nn.ModuleList([
            TransformerBlock_v1(d_model, num_heads, dim_feedforward, dropout) for _ in range(num_transformer_layers)
        ])
        self.color_node_blocks = nn.ModuleList([
            TransformerBlock_v1(d_model, num_heads, dim_feedforward, dropout) for _ in range(num_transformer_layers)
        ])

    def forward(self, node_embeddings, color_embeddings,
                node_self_mask=None, color_self_mask=None,
                node_to_color_mask=None, color_to_node_mask=None):
        x_nodes = node_embeddings
        x_colors = color_embeddings
        for layer_idx in range(self.num_layers):
            x_nodes = self.node_self_blocks[layer_idx](
                Q=x_nodes, K=x_nodes, V=x_nodes,
                attn_mask=node_self_mask
            )
            x_colors = self.color_self_blocks[layer_idx](
                Q=x_colors, K=x_colors, V=x_colors,
                attn_mask=color_self_mask
            )
            new_nodes = self.node_color_blocks[layer_idx](
                Q=x_nodes, K=x_colors, V=x_colors,
                attn_mask=node_to_color_mask
            )
            new_colors = self.color_node_blocks[layer_idx](
                Q=x_colors, K=new_nodes, V=new_nodes,
                attn_mask=color_to_node_mask
            )
            x_nodes = new_nodes
            x_colors = new_colors
        return x_nodes, x_colors
