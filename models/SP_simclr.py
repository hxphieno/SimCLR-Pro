import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
#
# class SPModel(nn.Module):
#     def __init__(self, num_classes):
#         super(SPModel, self).__init__()
#         self.resnet = models.resnet18(pretrained=False)
#         self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)
#
#         # 定义Transformer模型
#         encoder_layer = nn.TransformerEncoderLayer(d_model=num_classes, nhead=8, dim_feedforward=512, dropout=0.1)
#         self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=6)
#
#         self.conv1 = nn.Conv2d(in_channels=num_classes, out_channels=num_classes, kernel_size=1)
#
#     def forward(self, x):
#         x = self.resnet(x)
#         # 调整形状以匹配Transformer
#         x = x.unsqueeze(1)  # (batch_size, 1, num_classes)
#         x = x.permute(1, 0, 2)  # (1, batch_size, num_classes)
#         x = self.transformer(x)
#         x = x.permute(1, 0, 2).squeeze(1)  # 恢复形状为 (batch_size, num_classes)
#         x = x.unsqueeze(-1).unsqueeze(-1)  # (batch_size, num_classes, 1, 1)
#         x = self.conv1(x)
#         x = x.squeeze(-1).squeeze(-1)  # 恢复为 (batch_size, num_classes)
#         return x
#
#
# class SPSimCLR(nn.Module):
#     def __init__(self, out_dim):
#         super(SPSimCLR, self).__init__()
#         self.backbone = SPModel(num_classes=out_dim)
#
#         # 添加完整的MLP投影头
#         self.fc = nn.Sequential(
#             nn.Linear(out_dim, out_dim),
#             nn.ReLU(),
#             nn.Linear(out_dim, out_dim)
#         )
#
#     def forward(self, x):
#         x = self.backbone(x)
#         x = x.view(x.size(0), -1)  # 展平张量
#         x = self.fc(x)
#         return x


# class SPModel(nn.Module):
#     def __init__(self, num_classes):
#         super(SPModel, self).__init__()
#
#
#     def forward(self, x):
#
#         return x
#
#
# class SPSimCLR(nn.Module):
#     def __init__(self, out_dim):
#         super(SPSimCLR, self).__init__()
#         self.backbone = models.resnet18(pretrained=False, num_classes=out_dim)
#         dim_mlp = self.backbone.fc.in_features
#         self.backbone.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.backbone.fc)
#         encoder_layer = nn.TransformerEncoderLayer(d_model=out_dim, nhead=8, dim_feedforward=512, dropout=0.1)
#         self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=6)
#
#
#     def forward(self, x):
#         x = self.backbone(x)
#         x = x.unsqueeze(1)  # 变为 (batch_size, 1, out_dim)
#         # 通过 Transformer 编码器
#         x = self.transformer(x)
#         # 去掉序列维度，恢复为 (batch_size, out_dim)
#         x = x.squeeze(1)
#         return x


import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models.vision_transformer import vit_b_16

import torch.nn.functional as F

import torch.nn.functional as F


import torch
import torch.nn as nn
from torchvision.models import vit_b_16

class SPModel(nn.Module):
    def __init__(self, num_classes):
        super(SPModel, self).__init__()

        # 加载预训练的 ViT 模型
        self.vit = vit_b_16(pretrained=True)

        # 替换 ViT 的分类头
        if isinstance(self.vit.heads, nn.Sequential):
            in_features = self.vit.heads[0].in_features  # 访问 Sequential 中的第一个线性层
        else:
            in_features = self.vit.heads.in_features  # 如果是单独的线性层

        self.vit.heads = nn.Sequential(
            nn.Conv2d(in_features, 512, kernel_size=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Flatten(),  # 添加 Flatten 层将 4D 张量转换为 2D 张量
            nn.Linear(512, num_classes)
        )
        self.in_features=in_features


        self.se_block = SEBlock(num_classes)


    def forward(self, x):
        # 输入到 ViT
        x = self.vit(x)  # (batch_size, num_classes)
        x = self.se_block(x.unsqueeze(-1).unsqueeze(-1))
        x = x.squeeze(-1).squeeze(-1)
        return x


class SPSimCLR(nn.Module):
    def __init__(self, out_dim):
        super(SPSimCLR, self).__init__()
        self.backbone = SPModel(num_classes=out_dim)

        dim_mlp = self.backbone.in_features

        # 添加投影头 (MLP Projection Head)
        self.backbone.vit.heads = nn.Sequential(
            nn.Linear(dim_mlp, dim_mlp),
            nn.ReLU(),
            nn.Linear(dim_mlp, out_dim)
        )

    def forward(self, x):
        return self.backbone(x)

class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y