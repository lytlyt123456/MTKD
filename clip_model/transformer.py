import torch
from torch import nn
import torch.nn.functional as F
from attention_block import LayerNorm
from transformer_encoder import TransformerEncoder_IVLP
from transformer_encoder import TransformerEncoder_MaPLe

class Transformer_IVLP(nn.Module):
    def __init__(self, vocab_size, word_dim, block_num, num_heads,
                 prompt_depth, design_details, context_length, output_dim):
        super().__init__()
        conv0 = nn.Conv2d(1, 1, 1)
        self.dtype = conv0.weight.dtype
        self.text_embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=word_dim)
        self.ln1 = LayerNorm(normalized_shape=word_dim)
        self.ln2 = LayerNorm(normalized_shape=word_dim)
        self.prompt_length = design_details['prompt_length']
        self.context_length = context_length
        self.class_token = nn.Parameter(torch.randn(1, word_dim) / word_dim ** 0.5)
        self.positional_embedding = nn.Parameter(torch.randn(1 + context_length, word_dim) / word_dim ** 0.5)
        self.initial_prompt = nn.Parameter(torch.randn(self.prompt_length, word_dim) / word_dim ** 0.5)
        self.transformer_encoder = TransformerEncoder_IVLP(
            block_num=block_num, embed_dim=word_dim, num_heads=num_heads,
            prompt_depth=prompt_depth, is_text_layer=True, design_details=design_details,
            attn_mask=self.build_attn_mask()
        )
        self.proj = nn.Linear(word_dim, output_dim)

    def build_attn_mask(self):
        attn_mask = torch.empty(1 + self.prompt_length + self.context_length,
                                1 + self.prompt_length + self.context_length)
        attn_mask.fill_(float('-inf'))
        # [[-inf, -inf, -inf],
        #  [-inf, -inf, -inf],
        #  [-inf, -inf, -inf]]
        attn_mask.triu_(1)
        # [[0,    -inf, -inf],
        #  [0,    0,    -inf],
        #  [0,    0,    0   ]]
        return attn_mask

    def forward(self, text):
        x = self.text_embedding(text).type(self.dtype) # [B, N, C]
        x = x.permute(1, 0, 2) # [N, B, C]
        x = torch.cat([self.class_token.expand(x.shape[1], -1, -1).permute(1, 0, 2).type(x.dtype), x], dim=0) # [1+N, B, C]
        x = x + self.positional_embedding[ : , None, : ].type(x.dtype)
        x = torch.cat([
            x[ : 1, : , : ],
            self.initial_prompt.expand(x.shape[1], -1, -1).permute(1, 0, 2).type(x.dtype),
            x[1 : , : , : ]
        ], dim=0)
        x = self.ln1(x)
        x = self.transformer_encoder(x)
        x = x[0, : , : ]
        x = self.ln2(x)
        x = self.proj(x)
        return x

class Transformer_MaPLe(nn.Module):
    def __init__(self,vocab_size, word_dim, block_num, num_heads,
                 design_details, context_length, output_dim):
        super().__init__()
        conv0 = nn.Conv2d(1, 1, 1)
        self.dtype = conv0.weight.dtype
        self.text_embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=word_dim)
        self.ln1 = LayerNorm(normalized_shape=word_dim)
        self.ln2 = LayerNorm(normalized_shape=word_dim)
        self.prompt_length = design_details['prompt_length']
        self.context_length = context_length
        self.class_token = nn.Parameter(torch.randn(1, word_dim) / word_dim ** 0.5)
        self.positional_embedding = nn.Parameter(torch.randn(1 + context_length, word_dim) / word_dim ** 0.5)
        self.transformer_encoder = TransformerEncoder_MaPLe(
            embed_dim=word_dim, num_heads=num_heads, block_num=block_num,
            design_details=design_details, is_text_layer=True, attn_mask=self.build_attn_mask()
        )
        self.proj = nn.Linear(word_dim, output_dim)

    def build_attn_mask(self):
        attn_mask = torch.empty(1 + self.prompt_length + self.context_length,
                                1 + self.prompt_length + self.context_length)
        attn_mask.fill_(float('-inf'))
        # [[-inf, -inf, -inf],
        #  [-inf, -inf, -inf],
        #  [-inf, -inf, -inf]]
        attn_mask.triu_(1)
        # [[0   , -inf, -inf],
        #  [0   , 0   , -inf],
        #  [0   , 0   , 0   ]]
        return attn_mask

    def forward(self, text, initial_prompt, compound_prompt_deeper):
        x = self.text_embedding(text).type(self.dtype)  # [B, N, C]
        x = x.permute(1, 0, 2)  # [N, B, C]
        x = torch.cat([self.class_token.expand(x.shape[1], -1, -1).permute(1, 0, 2).type(x.dtype), x], dim=0)  # [1+N, B, C]
        x = x + self.positional_embedding[:, None, :].type(x.dtype)
        x = torch.cat([
            x[ : 1, : , : ],
            initial_prompt.expand(x.shape[1], -1, -1).permute(1, 0, 2).to(dtype=x.dtype, device=x.device),
            x[1 : , : , : ]
        ], dim=0) # [1+PL+N, B, C]
        x = self.ln1(x)
        x = self.transformer_encoder([x, compound_prompt_deeper, 0])[0]
        x = x[0, :, :]
        x = self.ln2(x)
        x = self.proj(x)
        return x
