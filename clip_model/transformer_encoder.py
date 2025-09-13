import torch
from torch import nn
import torch.nn.functional as F
from attention_block import ResidualAttentionBlock_IVLP
from attention_block import ResidualAttentionBlock_MaPLe

class TransformerEncoder_IVLP(nn.Module):
    def __init__(self, block_num, embed_dim, num_heads, prompt_depth, is_text_layer : bool, design_details, attn_mask = None):
        super().__init__()
        self.blocks = nn.Sequential(*[ResidualAttentionBlock_IVLP(
            embed_dim=embed_dim, num_heads=num_heads,
            block_seq_num=i, add_new_prompt = (i < prompt_depth),
            is_text_layer=is_text_layer, design_details=design_details,
            attn_mask=attn_mask
        ) for i in range(block_num)])

    def forward(self, x):
        return self.blocks(x)

class TransformerEncoder_MaPLe(nn.Module):
    def __init__(self, embed_dim, num_heads, block_num, design_details, is_text_layer : bool, attn_mask : torch.Tensor = None):
        super().__init__()
        self.blocks = nn.Sequential(*[ResidualAttentionBlock_MaPLe(
            embed_dim=embed_dim, num_heads=num_heads,
            design_details=design_details, block_seq_num=i,
            attn_mask=attn_mask, is_text_layer=is_text_layer
        ) for i in range(block_num)])

    def forward(self, x):
        return self.blocks(x)
