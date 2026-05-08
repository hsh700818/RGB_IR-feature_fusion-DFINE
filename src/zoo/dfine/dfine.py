import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...core import register

try:
    from torchvision.ops import DeformConv2d
except Exception:
    DeformConv2d = None

__all__ = ["DFINE"]


class GLSA(nn.Module):
    """
    Global-local spatial attention for small aerial targets.
    It keeps the original residual path and uses lightweight spatial/channel
    weights to refine fused RGB-IR features.
    """

    def __init__(self, ch):
        super().__init__()
        hidden_ch = max(ch // 8, 16)

        self.avg_pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.avg_pool_w = nn.AdaptiveAvgPool2d((1, None))

        self.conv_h = nn.Conv2d(ch, ch, kernel_size=(3, 1), padding=(1, 0), groups=ch)
        self.conv_w = nn.Conv2d(ch, ch, kernel_size=(1, 3), padding=(0, 1), groups=ch)

        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(ch, hidden_ch, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_ch, ch, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        identity = x
        x_h = self.conv_h(self.avg_pool_h(x))
        x_w = self.conv_w(self.avg_pool_w(x))
        x_c = self.channel_att(x)
        return identity * x_h.sigmoid() * x_w.sigmoid() * x_c


class AlignmentAwareFusion(nn.Module):
    """
    Residual deformable alignment for RGB-IR feature misalignment.

    The learned offset is bounded and initialized to zero. The deformable branch
    is attached through a zero-initialized residual scale, so replacing the old
    fusion block does not immediately disturb a pretrained or partially trained
    model.
    """

    def __init__(self, ch, max_offset=4.0):
        super().__init__()
        self.max_offset = float(max_offset)

        self.offset_conv = nn.Conv2d(ch * 2, 2 * 3 * 3, kernel_size=3, padding=1)
        self.mask_conv = nn.Conv2d(ch * 2, 1 * 3 * 3, kernel_size=3, padding=1)

        if DeformConv2d is not None:
            self.dcn = DeformConv2d(ch, ch, kernel_size=3, padding=1, bias=False)
            self.fallback_align = None
        else:
            self.dcn = None
            self.fallback_align = nn.Sequential(
                nn.Conv2d(ch, ch, kernel_size=3, padding=1, groups=ch, bias=False),
                nn.Conv2d(ch, ch, kernel_size=1, bias=False),
            )

        self.align_scale = nn.Parameter(torch.zeros(1))
        self.fusion_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(ch * 2, ch, 1),
            nn.Sigmoid(),
        )

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.constant_(self.offset_conv.weight, 0.0)
        nn.init.constant_(self.offset_conv.bias, 0.0)
        nn.init.constant_(self.mask_conv.weight, 0.0)
        nn.init.constant_(self.mask_conv.bias, 0.0)

    def forward(self, f_rgb, f_ir):
        feat_cat = torch.cat([f_rgb, f_ir], dim=1)

        if self.dcn is not None:
            offsets = torch.tanh(self.offset_conv(feat_cat)) * self.max_offset
            mask = torch.sigmoid(self.mask_conv(feat_cat))
            aligned_delta = self.dcn(f_ir, offsets, mask)
        else:
            aligned_delta = self.fallback_align(f_ir)

        f_ir_aligned = f_ir + self.align_scale * aligned_delta
        gate = self.fusion_gate(torch.cat([f_rgb, f_ir_aligned], dim=1))
        return f_rgb * gate + f_ir_aligned * (1.0 - gate)


class CrossModalInteraction(nn.Module):
    """
    Memory-safe cross-modal interaction.

    The Gemini-style full HW x HW attention is expensive on P3. This variant
    performs the cross-modal attention on adaptively pooled tokens and then
    upsamples the complementary context back to the original feature size.
    """

    def __init__(self, ch, reduction=4, max_tokens=196):
        super().__init__()
        attn_ch = max(ch // reduction, 32)
        self.max_tokens = int(max_tokens)

        self.query_conv = nn.Conv2d(ch, attn_ch, 1)
        self.key_conv = nn.Conv2d(ch, attn_ch, 1)
        self.value_conv = nn.Conv2d(ch, ch, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def _pool_if_needed(self, x):
        _, _, h, w = x.shape
        if h * w <= self.max_tokens:
            return x, (h, w)

        pooled_hw = max(int(self.max_tokens ** 0.5), 1)
        return F.adaptive_avg_pool2d(x, (pooled_hw, pooled_hw)), (pooled_hw, pooled_hw)

    def forward(self, f_main, f_aux):
        b, c, h, w = f_main.shape

        f_main_pool, _ = self._pool_if_needed(f_main)
        f_aux_pool, _ = self._pool_if_needed(f_aux)

        _, _, hp, wp = f_main_pool.shape
        token_num = hp * wp

        proj_query = self.query_conv(f_main_pool).flatten(2).transpose(1, 2)
        proj_key = self.key_conv(f_aux_pool).flatten(2)
        energy = torch.bmm(proj_query, proj_key) / (proj_key.shape[1] ** 0.5)
        attention = torch.softmax(energy, dim=-1)

        proj_value = self.value_conv(f_aux_pool).flatten(2)
        out = torch.bmm(proj_value, attention.transpose(1, 2))
        out = out.view(b, c, hp, wp)

        if token_num != h * w:
            out = F.interpolate(out, size=(h, w), mode="bilinear", align_corners=False)

        return f_main + self.gamma * out


class ScaleGate(nn.Module):
    """Per-scale learned RGB/IR contribution gate (DFS-lite).

    Produces two spatial weight maps w_rgb and w_ir from the concatenated
    RGB+IR features.  The maps are constrained to sum to 1 via softmax so the
    total feature energy is preserved.

    ir_bias shifts the initial gate toward IR (positive = favour IR at init).
    P3 (small targets, thermal contrast) benefits from ir_bias > 0; P5
    (large context, texture) benefits from ir_bias < 0 (favour RGB).
    """

    def __init__(self, ch, ir_bias: float = 0.0):
        super().__init__()
        hidden = max(ch // 8, 16)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(ch * 2, hidden, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 2, 1),  # 2 logits: [w_rgb, w_ir]
        )
        # Bias the last conv so w_ir starts higher when ir_bias > 0
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.constant_(self.gate[-1].bias, 0.0)
        with torch.no_grad():
            self.gate[-1].bias[0] -= ir_bias  # rgb logit
            self.gate[-1].bias[1] += ir_bias  # ir  logit

    def forward(self, f_rgb, f_ir):
        logits = self.gate(torch.cat([f_rgb, f_ir], dim=1))  # [B, 2, 1, 1]
        weights = torch.softmax(logits, dim=1)               # sums to 1
        w_rgb = weights[:, 0:1]
        w_ir  = weights[:, 1:2]
        return w_rgb, w_ir


class SimpleConcatFusion(nn.Module):
    """
    Plain RGB-IR concatenation fusion for the no-IACF ablation.

    This block keeps the dual-stream RGB/IR input pathway but removes the
    illumination estimation, cross-modal interaction, deformable alignment,
    scale gate, and GLSA components used by AdvancedMultimodalFusion.
    """

    def __init__(self, ch):
        super().__init__()
        self.fusion_conv = nn.Conv2d(ch * 2, ch, 1)

    def forward(self, f_rgb, f_ir):
        return self.fusion_conv(torch.cat([f_rgb, f_ir], dim=1))


class AdvancedMultimodalFusion(nn.Module):
    """
    AAF + optional CMI + scale-aware DFS-lite gate + optional GLSA.

    Ablation flags (all default True to reproduce the full model):
      use_cmi   — CrossModalInteraction on this level
      use_illum — kept for backward compat; when True the illumination scalar
                  is multiplied on top of the DFS-lite gate (additive signal)
      use_glsa  — Global-Local Spatial Attention on the fused feature

    ir_bias: initial bias toward IR in the scale gate.  Positive = favour IR.
    Caller sets this per FPN level (P3 → +1.0, P4 → 0.0, P5 → -1.0).
    """

    def __init__(self, ch, use_cmi=True, max_tokens=196, use_illum=True, use_glsa=True,
                 ir_bias: float = 0.0):
        super().__init__()
        self.use_cmi = bool(use_cmi)
        self.use_illum = bool(use_illum)
        self.use_glsa = bool(use_glsa)
        self.alignment = AlignmentAwareFusion(ch)

        if self.use_cmi:
            self.rgb_enhance = CrossModalInteraction(ch, max_tokens=max_tokens)
            self.ir_enhance = CrossModalInteraction(ch, max_tokens=max_tokens)
        else:
            self.rgb_enhance = nn.Identity()
            self.ir_enhance = nn.Identity()

        # DFS-lite: per-scale learned gate (always present)
        self.scale_gate = ScaleGate(ch, ir_bias=ir_bias)

        if self.use_illum:
            hidden_ch = max(ch // 4, 16)
            self.illum_extract = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(ch, hidden_ch, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden_ch, 1, 1),
                nn.Sigmoid(),
            )
        else:
            self.illum_extract = None

        self.fusion_conv = nn.Conv2d(ch * 2, ch, 1)

        if self.use_glsa:
            self.glsa = GLSA(ch)
        else:
            self.glsa = nn.Identity()

    def forward(self, f_rgb, f_ir):
        f_ir_aligned = self.alignment(f_rgb, f_ir)

        if self.use_cmi:
            f_rgb_ext = self.rgb_enhance(f_rgb, f_ir_aligned)
            f_ir_ext = self.ir_enhance(f_ir_aligned, f_rgb)
        else:
            f_rgb_ext = f_rgb
            f_ir_ext = f_ir_aligned

        # DFS-lite scale gate: per-level learned RGB/IR weights
        w_rgb, w_ir = self.scale_gate(f_rgb_ext, f_ir_ext)

        if self.use_illum and self.illum_extract is not None:
            # Illumination scalar modulates the IR weight further
            illum_score = self.illum_extract(f_rgb)  # bright→RGB, dark→IR
            w_ir = w_ir * (2.0 - illum_score)
            w_rgb = w_rgb * illum_score

        f_cat = torch.cat([f_rgb_ext * w_rgb, f_ir_ext * w_ir], dim=1)
        f_fused = self.fusion_conv(f_cat)
        return self.glsa(f_fused)


IlluminationAwareFusion = AdvancedMultimodalFusion


@register()
class DFINE(nn.Module):
    __inject__ = ["backbone", "encoder", "decoder"]

    def __init__(
        self,
        backbone: nn.Module,
        encoder: nn.Module,
        decoder: nn.Module,
        fusion_use_cmi: bool = True,
        fusion_use_illum: bool = True,
        fusion_use_glsa: bool = True,
        fusion_mode: str = "advanced",
    ):
        super().__init__()
        self.backbone = backbone
        self.backbone_ir = copy.deepcopy(backbone)
        self.encoder = encoder
        self.decoder = decoder
        self.fusion_mode = str(fusion_mode).lower()

        supported_fusion_modes = {"advanced", "iacf", "simple_concat", "concat"}
        if self.fusion_mode not in supported_fusion_modes:
            raise ValueError(
                f"Unsupported fusion_mode={fusion_mode}. "
                f"Expected one of {sorted(supported_fusion_modes)}."
            )

        in_channels = encoder.in_channels
        self.fusion_layers = nn.ModuleList()
        # ir_bias per FPN level: P3 (small targets, thermal) → +1.0,
        # P4 (medium) → 0.0, P5 (large context, texture) → -1.0
        num_levels = len(in_channels)
        ir_biases = [1.0 - 2.0 * i / max(num_levels - 1, 1) for i in range(num_levels)]
        for level_idx, (ch, ir_bias) in enumerate(zip(in_channels, ir_biases)):
            if self.fusion_mode in {"advanced", "iacf"}:
                # P3 (level 0) is the largest map — skip CMI there for speed.
                use_cmi_level = fusion_use_cmi and (level_idx > 0)
                fusion_layer = AdvancedMultimodalFusion(
                    ch,
                    use_cmi=use_cmi_level,
                    max_tokens=196,
                    use_illum=fusion_use_illum,
                    use_glsa=fusion_use_glsa,
                    ir_bias=ir_bias,
                )
            else:
                fusion_layer = SimpleConcatFusion(ch)

            self.fusion_layers.append(fusion_layer)

    def forward(self, x, targets=None):
        x_rgb = x[:, :3, :, :]
        x_ir = x[:, 3:, :, :]

        feats_rgb = self.backbone(x_rgb)
        feats_ir = self.backbone_ir(x_ir)

        fused_feats = []
        for f_rgb, f_ir, fusion in zip(feats_rgb, feats_ir, self.fusion_layers):
            fused_feats.append(fusion(f_rgb, f_ir))

        x = self.encoder(fused_feats)
        x = self.decoder(x, targets)
        return x

    def deploy(self):
        self.eval()
        for m in self.modules():
            if hasattr(m, "convert_to_deploy"):
                m.convert_to_deploy()
        return self
