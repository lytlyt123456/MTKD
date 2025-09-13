import torch
from torch import nn
import torch.nn.functional as F
from transformer_encoder import TransformerEncoder_IVLP
from transformer_encoder import TransformerEncoder_MaPLe
from attention_block import LayerNorm

class VisionTransformer_IVLP(nn.Module):
    def __init__(self, embed_dim, output_dim, patch_size, block_num, num_heads,
                 prompt_depth, design_details, input_size=224):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=embed_dim,
                               kernel_size=patch_size, stride=patch_size,
                               padding=0, bias=False)
        self.class_token = nn.Parameter(torch.randn(1, embed_dim) / embed_dim ** 0.5)
        self.positional_embedding = nn.Parameter(
            torch.randn(1 + (input_size // patch_size) ** 2, embed_dim) / embed_dim ** 0.5)
        self.transformer = TransformerEncoder_IVLP(
            block_num=block_num, embed_dim=embed_dim, num_heads=num_heads,
            prompt_depth=prompt_depth, is_text_layer = False,
            design_details = design_details, attn_mask=None
        )
        self.ln1 = LayerNorm(embed_dim)
        self.ln2 = LayerNorm(embed_dim)
        self.proj = nn.Linear(embed_dim, output_dim)
        self.prompt_length = design_details['prompt_length']
        self.initial_prompt = nn.Parameter(torch.randn(self.prompt_length, embed_dim) / embed_dim ** 0.5)

    def forward(self, x):
        x = x.type(self.conv1.weight.dtype)
        x = self.conv1(x) # [B, C, H, W]
        x = x.flatten(start_dim=2).permute(2, 0, 1) # [HW, B, C]
        x = torch.cat([self.class_token.expand(x.shape[1], -1, -1).permute(1, 0, 2).type(x.dtype), x], dim=0) # [1+HW, B, C]
        x = x + self.positional_embedding[ : , None, : ].type(x.dtype)
        x = torch.cat([x, self.initial_prompt.expand(x.shape[1], -1, -1).permute(1, 0, 2).type(x.dtype)], dim=0) # [1+HW+PL, B, C]
        x = self.ln1(x)
        x = self.transformer(x)
        x = x[0, : , : ]
        x = self.ln2(x)
        x = self.proj(x)
        return x

class VisionTransformer_MaPLe(nn.Module):
    def __init__(self, embed_dim, output_dim, patch_size, num_heads,
                 block_num, design_details, input_size = 224):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=embed_dim,
                               kernel_size=patch_size, stride=patch_size,
                               padding=0, bias=False)
        self.class_token = nn.Parameter(torch.randn(1, embed_dim) / embed_dim ** 0.5)
        self.positional_embedding = nn.Parameter(torch.randn(
            1 + (input_size // patch_size) ** 2, embed_dim
        ) / embed_dim ** 0.5)
        self.ln1 = LayerNorm(normalized_shape=embed_dim)
        self.ln2 = LayerNorm(normalized_shape=embed_dim)
        self.transformer = TransformerEncoder_MaPLe(embed_dim=embed_dim, num_heads=num_heads,
                                       block_num=block_num, design_details=design_details,
                                       is_text_layer=False, attn_mask=None)
        self.proj = nn.Linear(embed_dim, output_dim)

    def forward(self, x, initial_prompt, compound_prompt_deeper):
        x = x.type(self.conv1.weight.dtype)
        x = self.conv1(x) # [B, C, H, W]
        x = x.flatten(start_dim=2).permute(2, 0, 1) # [HW, B, C]
        x = torch.cat([self.class_token.expand(x.shape[1], -1, -1).permute(1, 0, 2).type(x.dtype), x], dim=0) # [1+HW, B, C]
        x = x + self.positional_embedding[ : , None, : ].type(x.dtype) # [HW+1, B, C]

        # initial_prompt: [prompt_length, C]
        x = torch.cat([
            x, initial_prompt.expand(x.shape[1], -1, -1).permute(1, 0, 2).to(dtype=x.dtype, device=x.device)
        ], dim=0) # [1+HW+PL, B, C]

        x = self.ln1(x)
        x = self.transformer([x, compound_prompt_deeper, 0])[0]
        x = x[0, : , : ]
        x = self.ln2(x)
        x = self.proj(x)
        return x
