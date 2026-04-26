import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from ...core import register


class GLSA(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.avg_pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.avg_pool_w = nn.AdaptiveAvgPool2d((1, None))
        
        self.conv_h = nn.Conv2d(ch, ch, kernel_size=(3, 1), padding=(1, 0), groups=ch)
        self.conv_w = nn.Conv2d(ch, ch, kernel_size=(1, 3), padding=(0, 1), groups=ch)
        
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(ch, ch // 8, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch // 8, ch, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        identity = x
        x_h = self.conv_h(self.avg_pool_h(x))
        x_w = self.conv_w(self.avg_pool_w(x))
        x_c = self.channel_att(x)
        return identity * x_h.sigmoid() * x_w.sigmoid() * x_c

class IlluminationAwareFusion(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.illum_extract = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(ch, ch // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch // 4, 1, 1),
            nn.Sigmoid()
        )
        self.fusion_conv = nn.Conv2d(ch * 2, ch, 1)
        self.glsa = GLSA(ch)

    def forward(self, f_rgb, f_ir):
        illum_score = self.illum_extract(f_rgb) 
        # 动态权重：illum_score 越小（夜间），IR 特征 f_ir 占比越高
        f_cat = torch.cat([f_rgb * illum_score, f_ir * (2.0 - illum_score)], dim=1)
        f_fused = self.fusion_conv(f_cat)
        return self.glsa(f_fused)

@register()
class DFINE(nn.Module):
    __inject__ = ["backbone", "encoder", "decoder"]

    def __init__(self, backbone: nn.Module, encoder: nn.Module, decoder: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.backbone_ir = copy.deepcopy(backbone)
        self.encoder = encoder
        self.decoder = decoder
        
        # 自动识别 YML 传入的层数 (S 版本为 3 层 [256, 512, 1024])
        in_channels = encoder.in_channels 
        
        self.fusion_layers = nn.ModuleList([
            IlluminationAwareFusion(ch) for ch in in_channels
        ])

    def forward(self, x, targets=None):
        # x: [Batch, 6, H, W]
        x_rgb = x[:, :3, :, :]
        x_ir = x[:, 3:, :, :]

        feats_rgb = self.backbone(x_rgb)
        feats_ir = self.backbone_ir(x_ir)

        fused = []
        for i, (f_rgb, f_ir) in enumerate(zip(feats_rgb, feats_ir)):
            # 自动适配 P3, P4, P5 的融合
            f_fused = self.fusion_layers[i](f_rgb, f_ir)
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