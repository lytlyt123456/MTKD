import torch
from torch import nn
import torch.nn.functional as F

class LayerNorm(nn.LayerNorm):
    def forward(self, x):
        t = x.dtype
        out = super().forward(x.type(torch.float32))
        return out.type(t)

class QuickGELU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(1.702 * x)

class ResidualAttentionBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, attn_mask : torch.Tensor=None):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads)
        self.ln1 = LayerNorm(normalized_shape=embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            QuickGELU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )
        self.ln2 = LayerNorm(normalized_shape=embed_dim)
        self.attn_mask = attn_mask

    def attention(self, x):
        if self.attn_mask is not None:
            self.attn_mask = self.attn_mask.to(device=x.device, dtype=x.dtype)
        return self.attn(query=x, key=x, value=x, need_weights=False, attn_mask=self.attn_mask)

    def forward(self, x):
        x = x + self.attention(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class ResidualAttentionBlock_IVLP(nn.Module):
    def __init__(self, embed_dim, num_heads, block_seq_num, add_new_prompt : bool, is_text_layer : bool, design_details, attn_mask=None):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads)
        self.ln1 = LayerNorm(normalized_shape=embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            QuickGELU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )
        self.ln2 = LayerNorm(normalized_shape=embed_dim)
        self.attn_mask = attn_mask
        self.is_first_layer = (block_seq_num == 0)
        self.add_new_prompt = add_new_prompt
        self.is_text_layer = is_text_layer
        self.prompt_length = design_details['prompt_length']
        self.prompt = nn.Parameter(torch.randn(self.prompt_length, embed_dim) / embed_dim ** 0.5)

    def attention(self, x):
        if self.attn_mask is not None:
            self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device)
        return self.attn(query=x, key=x, value=x, need_weights=False, attn_mask=self.attn_mask)

    def forward(self, x):
        if not self.is_first_layer and self.add_new_prompt:
            if not self.is_text_layer:
                prefix = x[ : x.shape[0] - self.prompt_length, : , : ]
                x = torch.cat([prefix, self.prompt.expand(x.shape[1], -1, -1).permute(1, 0, 2).type(x.dtype)], dim=0)
            else:
                prefix = x[ : 1, : , : ]
                suffix = x[1 + self.prompt_length : , : , : ]
                x = torch.cat([prefix, self.prompt.expand(x.shape[1], -1, -1).permute(1, 0, 2).type(x.dtype), suffix], dim=0)
        x = x + self.attention(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class ResidualAttentionBlock_MaPLe(nn.Module):
    def __init__(self, embed_dim, num_heads, design_details, block_seq_num = 0,
                 attn_mask : torch.Tensor = None, is_text_layer = False):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads)
        self.ln1 = LayerNorm(normalized_shape=embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            QuickGELU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )
        self.ln2 = LayerNorm(normalized_shape=embed_dim)
        self.attn_mask = attn_mask

        self.first_layer = (block_seq_num == 0)
        self.prompt_length = design_details['prompt_length']
        self.is_text_layer = is_text_layer

    def attention(self, x):
        if self.attn_mask is not None:
            self.attn_mask = self.attn_mask.to(device=x.device, dtype=x.dtype)
        return self.attn(query=x, key=x, value=x, need_weights=False, attn_mask=self.attn_mask)

    def forward(self, inputs):
        x = inputs[0]
        compound_prompt_deeper = inputs[1]
        counter = inputs[2]

        if not self.first_layer and counter < len(compound_prompt_deeper):
            if not self.is_text_layer:
                prefix = x[ : x.shape[0] - self.prompt_length, : , : ]
                vision_prompt = compound_prompt_deeper[counter].expand(x.shape[1], -1, -1).permute(1, 0, 2).to(dtype=x.dtype, device=x.device)
                x = torch.cat([prefix, vision_prompt], dim=0)
                counter += 1
            else:
                prefix = x[ : 1, : , : ]
                text_prompt = compound_prompt_deeper[counter].expand(x.shape[1], -1, -1).permute(1, 0, 2).to(dtype=x.dtype, device=x.device)
                suffix = x[1 + self.prompt_length : , : , : ]
                x = torch.cat([prefix, text_prompt, suffix], dim=0)
                counter += 1

        x = x + self.attention(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return [x, compound_prompt_deeper, counter]
