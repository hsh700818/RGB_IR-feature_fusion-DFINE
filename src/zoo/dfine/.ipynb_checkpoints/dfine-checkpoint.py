import torch
import torch.nn as nn
import copy
from ...core import register

class AdaptiveFusionModule(nn.Module):
    def __init__(self, ch):
        super().__init__()
        # 参考 CMAI-Det 的自适应思想 
        # 计算 RGB 和 IR 的权重分配
        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(ch * 2, ch // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch // 4, 2, 1),
            nn.Softmax(dim=1) # 生成两个模态的权重
        )
        self.conv = nn.Conv2d(ch, ch, kernel_size=3, padding=1, groups=ch)

    def forward(self, f_rgb, f_ir):
        cat = torch.cat([f_rgb, f_ir], dim=1)
        w = self.fc(self.pooling(cat)) # [B, 2, 1, 1]
        
        # 动态加权融合
        fused = f_rgb * w[:, 0:1, ...] + f_ir * w[:, 1:2, ...]
        return self.conv(fused)

@register()
class DFINE(nn.Module):
    __inject__ = ["backbone", "encoder", "decoder"]

    def __init__(self, backbone: nn.Module, encoder: nn.Module, decoder: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.backbone_ir = copy.deepcopy(backbone)
        self.encoder = encoder
        self.decoder = decoder
        
        in_channels = encoder.in_channels # 通常为 [256, 512, 1024]
        
        self.fusion_layers = nn.ModuleList([
            AdaptiveFusionModule(ch) for ch in in_channels
        ])

        # 使用空洞卷积捕捉小目标的上下文，防止被噪声淹没
        self.small_object_enhancer = nn.Sequential(
            nn.Conv2d(in_channels[0], in_channels[0], 3, padding=2, dilation=2),
            nn.BatchNorm2d(in_channels[0]),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, targets=None):
        # x: [Batch, 6, H, W]
        x_rgb = x[:, :3, :, :]
        x_ir = x[:, 3:, :, :]

        feats_rgb = self.backbone(x_rgb)
        feats_ir = self.backbone_ir(x_ir)

        fused = []
        for i, (f_rgb, f_ir) in enumerate(zip(feats_rgb, feats_ir)):
            # 1. 自适应权重融合 (参考 CMAI-Det) 
            f_fused = self.fusion_layers[i](f_rgb, f_ir)
            
            # 2. 小目标增强 (参考 IM-CMDet, 仅对 P3 层做) 
            if i == 0: 
                f_fused = f_fused + self.small_object_enhancer(f_fused)
            
            fused.append(f_fused)

        x = self.encoder(fused)
        x = self.decoder(x, targets)
        return x

    def deploy(self):
        self.eval()
        for m in self.modules():
            if hasattr(m, "convert_to_deploy"):
                m.convert_to_deploy()
        return self