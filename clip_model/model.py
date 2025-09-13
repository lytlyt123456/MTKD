import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from transformer import Transformer_IVLP
from transformer import Transformer_MaPLe
from vision_transformer import VisionTransformer_IVLP
from vision_transformer import VisionTransformer_MaPLe
from resnet import ModifiedResNet

class CLIP(nn.Module):
    def __init__(self, design_details,
                 # text_encoder & image_encoder:
                 block_num, feature_dim,
                 # text_encoder:
                 vocab_size, word_dim, context_length,
                 # image_encoder:
                 input_size,
                 # ResNet:
                 layer_nums, width,
                 # VisionTransformer:
                 embed_dim, patch_size
                 ):
        super().__init__()
        self.method = design_details['method'] # 'IVLP', 'MaPLe'
        self.image_encoder_net = design_details['image_encoder_net'] # 'ResNet', 'VisionTransformer'
        self.prompt_length = design_details['prompt_length']
        self.prompt_depth = design_details['prompt_depth']
        self.word_dim = word_dim
        self.embed_dim = embed_dim
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1.0 / 0.07))
        if self.image_encoder_net == 'ResNet':
            self.image_encoder = ModifiedResNet(
                layer_nums=layer_nums, num_heads=width * 32 // 64,
                output_dim=feature_dim, input_size=input_size, width=width
            )
            self.text_encoder = Transformer_IVLP(
                vocab_size=vocab_size, word_dim=word_dim, block_num=block_num,
                num_heads=word_dim // 64, prompt_depth=self.prompt_depth, design_details=design_details,
                context_length=context_length, output_dim=feature_dim
            )
        else: # image_encoder_net == 'VisionTransformer'
            if self.method == 'IVLP':
                self.text_encoder = Transformer_IVLP(
                    vocab_size=vocab_size, word_dim=word_dim, block_num=block_num,
                    num_heads=word_dim // 64, prompt_depth=self.prompt_depth, design_details=design_details,
                    context_length=context_length, output_dim=feature_dim
                )
                self.image_encoder = VisionTransformer_IVLP(
                    embed_dim=embed_dim, output_dim=feature_dim, patch_size=patch_size,
                    block_num=block_num, num_heads=embed_dim // 64, prompt_depth=self.prompt_depth,
                    design_details=design_details, input_size=input_size
                )
            else:
                self.text_encoder = Transformer_MaPLe(
                    vocab_size=vocab_size, word_dim=word_dim, block_num=block_num,
                    num_heads=word_dim // 64, design_details=design_details,
                    context_length=context_length, output_dim=feature_dim
                )
                self.image_encoder = VisionTransformer_MaPLe(
                    embed_dim=embed_dim, output_dim=feature_dim, patch_size=patch_size,
                    block_num=block_num, num_heads=embed_dim // 64,
                    design_details=design_details, input_size=input_size
                )
        self.text_initial_prompt = self.build_text_prompt()
        self.initial_prompt_proj = self.build_prompt_proj()
        self.compound_prompt_deeper_text = nn.ParameterList()
        self.compound_prompt_deeper_proj = nn.ModuleList()
        for _ in range(self.prompt_depth - 1):
            text_prompt = self.build_text_prompt()
            prompt_proj = self.build_prompt_proj()
            self.compound_prompt_deeper_text.append(text_prompt)
            self.compound_prompt_deeper_proj.append(prompt_proj)

    def build_text_prompt(self):
        return nn.Parameter(torch.randn(self.prompt_length, self.word_dim) / self.word_dim ** 0.5)

    def build_prompt_proj(self):
        return nn.Linear(self.word_dim, self.embed_dim, bias=False)

    def forward(self, images, texts):
        image_features = None
        text_features = None
        if self.image_encoder_net == 'ResNet' or self.method == 'IVLP':
            image_features = self.image_encoder(images)
            text_features = self.text_encoder(texts)
        else:
            text_features = self.text_encoder(texts, self.text_initial_prompt, self.compound_prompt_deeper_text)
            image_features = self.image_encoder(
                images,
                self.initial_prompt_proj(self.text_initial_prompt),
                nn.ParameterList([self.compound_prompt_deeper_proj[i](self.compound_prompt_deeper_text[i])
                                  for i in range(self.prompt_depth - 1)])
            )

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp().type(image_features.dtype)
        logits_per_image = image_features @ text_features.t() * logit_scale
        logits_per_text = text_features @ image_features.t() * logit_scale

        return logits_per_image, logits_per_text
