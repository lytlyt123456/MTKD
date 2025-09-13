import torch
from torch import nn
import torch.nn.functional as F

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, avg_pool_kernel_size):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=inplanes, out_channels=planes,
                               kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=planes)
        self.conv2 = nn.Conv2d(in_channels=planes, out_channels=planes,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_features=planes)
        self.conv3 = nn.Conv2d(in_channels=planes, out_channels=planes * Bottleneck.expansion,
                               kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(num_features=planes * Bottleneck.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.avg_pool = nn.AvgPool2d(kernel_size=avg_pool_kernel_size) if avg_pool_kernel_size > 1 else nn.Identity()

        self.down_sample = None
        if avg_pool_kernel_size > 1 or inplanes != planes * Bottleneck.expansion:
            self.down_sample = nn.Sequential(
                nn.AvgPool2d(avg_pool_kernel_size),
                nn.Conv2d(in_channels=inplanes, out_channels=planes * Bottleneck.expansion,
                          kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(num_features=planes * Bottleneck.expansion)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.avg_pool(out)
        out = self.bn3(self.conv3(out))

        identity = x
        if self.down_sample is not None:
            identity = self.down_sample(x)

        out += identity
        out = self.relu(out)

        return out

class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim, embed_dim, num_heads, output_dim=None):
        super().__init__()

        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads=num_heads

    def forward(self, x): # [B, C, H, W]
        x = x.flatten(start_dim=2).permute(2, 0, 1) # [HW, B, C]
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0) # [HW+1, B, C]
        x = x + self.positional_embedding[:, None, :].to(x.dtype)
        x, _ = F.multi_head_attention_forward(
            query=x, key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias], dim=0),
            bias_k=None, bias_v=None,
            add_zero_attn=False,
            dropout_p=0, out_proj_weight=self.c_proj.weight, out_proj_bias=self.c_proj.bias, training=True,
            use_separate_proj_weight=True,
            q_proj_weight=self.q_proj.weight, k_proj_weight=self.k_proj.weight, v_proj_weight=self.v_proj.weight,
            need_weights=False
        )
        return x[0]

class ModifiedResNet(nn.Module):
    def __init__(self, layer_nums, num_heads, output_dim, input_size=224, width=64):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width // 2,
                               kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.conv2 = nn.Conv2d(in_channels=width // 2, out_channels=width // 2,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.conv3 = nn.Conv2d(in_channels=width // 2, out_channels=width,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.relu = nn.ReLU(inplace=True)
        self.avg_pool = nn.AvgPool2d(kernel_size=2)
        self.inplanes = width

        self.layer1 = self.make_layer(width, 1, layer_nums[0])
        self.layer2 = self.make_layer(width * 2, 2, layer_nums[1])
        self.layer3 = self.make_layer(width * 4, 2, layer_nums[2])
        self.layer4 = self.make_layer(width * 8, 2, layer_nums[3])

        self.attn_pool = AttentionPool2d(input_size // 32, width * 32, num_heads, output_dim)

    def make_layer(self, planes, avg_pool_kernel_size, layer_num):
        layer = [Bottleneck(self.inplanes, planes, avg_pool_kernel_size)]
        self.inplanes = planes * Bottleneck.expansion
        for _ in range(layer_num - 1):
            layer.append(Bottleneck(self.inplanes, planes, 1))
        return nn.Sequential(*layer)

    def forward(self, x):
        x = x.type(self.conv1.weight.dtype)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.avg_pool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.attn_pool(x)
        return x
