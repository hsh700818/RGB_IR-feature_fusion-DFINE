import torch
import torch.nn as nn
import copy
from ...core import register

@register()
class DFINE(nn.Module):
    __inject__ = ["backbone", "encoder", "decoder"]

    def __init__(self, backbone: nn.Module, encoder: nn.Module, decoder: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.backbone_ir = copy.deepcopy(backbone) # 复制骨干网络
        self.encoder = encoder
        self.decoder = decoder
        
        # 自动获取通道数建立融合层
        in_channels = encoder.in_channels
        self.fusion_layers = nn.ModuleList([
            nn.Conv2d(ch * 2, ch, kernel_size=1) for ch in in_channels
        ])

    def forward(self, x, targets=None):
        # x: [Batch, 6, H, W]
        # 1. 切分 RGB 和 IR 通道
        x_rgb = x[:, :3, :, :]
        x_ir = x[:, 3:, :, :]

        # 2. 双流特征提取
        feats_rgb = self.backbone(x_rgb)
        feats_ir = self.backbone_ir(x_ir)

        # 3. 特征融合
        fused = []
        for i, (f_rgb, f_ir) in enumerate(zip(feats_rgb, feats_ir)):
            f_cat = torch.cat([f_rgb, f_ir], dim=1)
            fused.append(self.fusion_layers[i](f_cat))

        # 4. 传入编码器和解码器
        x = self.encoder(fused)
        x = self.decoder(x, targets)
        return x

    def deploy(self):
        self.eval()
        for m in self.modules():
            if hasattr(m, "convert_to_deploy"):
                m.convert_to_deploy()
        return self