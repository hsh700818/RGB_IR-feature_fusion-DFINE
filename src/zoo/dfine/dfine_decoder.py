import math
from collections import OrderedDict
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from ...core import register
from .utils import get_activation, inverse_sigmoid

__all__ = ["DFINETransformer"]


def bias_init_with_prob(prior_prob=0.01):
    return float(-math.log((1 - prior_prob) / prior_prob))


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, act="relu"):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.act = get_activation(act)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = self.act(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class TransformerDecoderLayer(nn.Module):
    """A practical DETR-style decoder layer for RGB-IR detection."""

    def __init__(
        self,
        hidden_dim=256,
        nhead=8,
        dim_feedforward=1024,
        dropout=0.0,
        activation="relu",
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(hidden_dim, nhead, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(hidden_dim, nhead, dropout=dropout, batch_first=True)

        self.linear1 = nn.Linear(hidden_dim, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, hidden_dim)
        self.activation = get_activation(activation)

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.dropout4 = nn.Dropout(dropout)

        self._reset_parameters()

    def _reset_parameters(self):
        init.xavier_uniform_(self.linear1.weight)
        init.xavier_uniform_(self.linear2.weight)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, tgt, memory, query_pos=None, memory_pos=None, attn_mask=None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2, _ = self.self_attn(q, k, value=tgt, attn_mask=attn_mask, need_weights=False)
        tgt = self.norm1(tgt + self.dropout1(tgt2))

        q = self.with_pos_embed(tgt, query_pos)
        k = self.with_pos_embed(memory, memory_pos)
        tgt2, _ = self.cross_attn(q, k, value=memory, need_weights=False)
        tgt = self.norm2(tgt + self.dropout2(tgt2))

        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = self.norm3(tgt + self.dropout4(tgt2))
        return tgt


@register()
class DFINETransformer(nn.Module):
    """Multi-scale transformer decoder.

    box_format controls the detection head:
      - "obb": predict [cx, cy, w, h, angle] rotated boxes.
      - "hbb": predict [cx, cy, w, h] horizontal boxes and do not build angle heads.
    """

    __share__ = ["num_classes", "eval_spatial_size"]

    def __init__(
        self,
        num_classes=5,
        hidden_dim=256,
        num_queries=300,
        feat_channels=[256, 256, 256],
        feat_strides=[8, 16, 32],
        num_levels=3,
        num_points=4,
        nhead=8,
        num_layers=6,
        dim_feedforward=1024,
        dropout=0.0,
        activation="relu",
        num_denoising=0,
        label_noise_ratio=0.5,
        box_noise_scale=1.0,
        learn_query_content=False,
        eval_spatial_size=None,
        eval_idx=-1,
        eps=1e-2,
        aux_loss=True,
        cross_attn_method="default",
        query_select_method="default",
        reg_max=32,
        reg_scale=4.0,
        layer_scale=1,
        num_bins=None,
        box_format="obb",
        **kwargs,
    ):
        super().__init__()
        assert len(feat_channels) <= num_levels
        assert len(feat_strides) == len(feat_channels)

        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.num_levels = num_levels
        self.num_layers = num_layers
        self.aux_loss = aux_loss
        self.eval_idx = eval_idx if eval_idx >= 0 else num_layers + eval_idx
        self.query_select_method = query_select_method
        self.learn_query_content = learn_query_content
        self.eval_spatial_size = eval_spatial_size
        self.feat_strides = feat_strides
        self.eps = eps
        self.box_format = str(box_format).lower()
        if self.box_format not in {"obb", "hbb"}:
            raise ValueError(f"Unsupported box_format={box_format}; expected 'obb' or 'hbb'.")
        self.predict_angle = self.box_format == "obb"

        self._build_input_proj_layer(feat_channels)
        self.level_embed = nn.Parameter(torch.Tensor(num_levels, hidden_dim))

        self.enc_output = nn.Sequential(
            OrderedDict([
                ("proj", nn.Linear(hidden_dim, hidden_dim)),
                ("norm", nn.LayerNorm(hidden_dim)),
            ])
        )
        self.enc_score_head = nn.Linear(hidden_dim, num_classes)
        self.enc_bbox_head = MLP(hidden_dim, hidden_dim, 4, 3)

        if learn_query_content:
            self.tgt_embed = nn.Embedding(num_queries, hidden_dim)
        self.query_pos_head = MLP(4, 2 * hidden_dim, hidden_dim, 2)

        self.layers = nn.ModuleList([
            TransformerDecoderLayer(hidden_dim, nhead, dim_feedforward, dropout, activation)
            for _ in range(num_layers)
        ])
        self.dec_score_head = nn.ModuleList([
            nn.Linear(hidden_dim, num_classes) for _ in range(num_layers)
        ])
        self.dec_bbox_head = nn.ModuleList([
            MLP(hidden_dim, hidden_dim, 4, 3) for _ in range(num_layers)
        ])
        if self.predict_angle:
            self.dec_angle_head = nn.ModuleList([
                MLP(hidden_dim, hidden_dim, 1, 3) for _ in range(num_layers)
            ])
            self.pre_angle_head = MLP(hidden_dim, hidden_dim, 1, 3)
        else:
            self.dec_angle_head = None
            self.pre_angle_head = None
        self.pre_bbox_head = MLP(hidden_dim, hidden_dim, 4, 3)

        self.num_denoising = 0
        self._reset_parameters(feat_channels)

    def _reset_parameters(self, feat_channels):
        bias = bias_init_with_prob(0.01)
        init.constant_(self.enc_score_head.bias, bias)
        init.constant_(self.enc_bbox_head.layers[-1].weight, 0)
        init.constant_(self.enc_bbox_head.layers[-1].bias, 0)
        init.constant_(self.pre_bbox_head.layers[-1].weight, 0)
        init.constant_(self.pre_bbox_head.layers[-1].bias, 0)

        if self.predict_angle:
            init.constant_(self.pre_angle_head.layers[-1].weight, 0)
            init.constant_(self.pre_angle_head.layers[-1].bias, 0)

        for i, (cls_, box_) in enumerate(zip(self.dec_score_head, self.dec_bbox_head)):
            init.constant_(cls_.bias, bias)
            init.constant_(box_.layers[-1].weight, 0)
            init.constant_(box_.layers[-1].bias, 0)
            if self.predict_angle:
                ang_ = self.dec_angle_head[i]
                init.constant_(ang_.layers[-1].weight, 0)
                init.constant_(ang_.layers[-1].bias, 0)

        init.xavier_uniform_(self.enc_output[0].weight)
        init.normal_(self.level_embed)
        if self.learn_query_content:
            init.xavier_uniform_(self.tgt_embed.weight)
        init.xavier_uniform_(self.query_pos_head.layers[0].weight)
        init.xavier_uniform_(self.query_pos_head.layers[1].weight)
        for m, in_channels in zip(self.input_proj, feat_channels):
            if in_channels != self.hidden_dim and hasattr(m, "__getitem__"):
                init.xavier_uniform_(m[0].weight)

    def _build_input_proj_layer(self, feat_channels):
        self.input_proj = nn.ModuleList()
        for in_channels in feat_channels:
            if in_channels == self.hidden_dim:
                self.input_proj.append(nn.Identity())
            else:
                self.input_proj.append(
                    nn.Sequential(OrderedDict([
                        ("conv", nn.Conv2d(in_channels, self.hidden_dim, 1, bias=False)),
                        ("norm", nn.BatchNorm2d(self.hidden_dim)),
                    ]))
                )

        in_channels = feat_channels[-1]
        for _ in range(self.num_levels - len(feat_channels)):
            if in_channels == self.hidden_dim:
                self.input_proj.append(nn.Identity())
            else:
                self.input_proj.append(
                    nn.Sequential(OrderedDict([
                        ("conv", nn.Conv2d(in_channels, self.hidden_dim, 3, 2, padding=1, bias=False)),
                        ("norm", nn.BatchNorm2d(self.hidden_dim)),
                    ]))
                )
                in_channels = self.hidden_dim

    @staticmethod
    def build_2d_sincos_position_embedding(w, h, embed_dim=256, temperature=10000.0, device=None):
        grid_w = torch.arange(int(w), dtype=torch.float32, device=device)
        grid_h = torch.arange(int(h), dtype=torch.float32, device=device)
        grid_w, grid_h = torch.meshgrid(grid_w, grid_h, indexing="ij")
        assert embed_dim % 4 == 0
        pos_dim = embed_dim // 4
        omega = torch.arange(pos_dim, dtype=torch.float32, device=device) / pos_dim
        omega = 1.0 / (temperature ** omega)
        out_w = grid_w.flatten()[..., None] @ omega[None]
        out_h = grid_h.flatten()[..., None] @ omega[None]
        return torch.cat([out_w.sin(), out_w.cos(), out_h.sin(), out_h.cos()], dim=1)[None]

    def _get_encoder_input(self, feats: List[torch.Tensor]):
        proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(feats)]
        if self.num_levels > len(proj_feats):
            len_srcs = len(proj_feats)
            for i in range(len_srcs, self.num_levels):
                if i == len_srcs:
                    proj_feats.append(self.input_proj[i](feats[-1]))
                else:
                    proj_feats.append(self.input_proj[i](proj_feats[-1]))

        feat_flatten, pos_flatten, spatial_shapes = [], [], []
        for i, feat in enumerate(proj_feats):
            b, c, h, w = feat.shape
            spatial_shapes.append([h, w])
            feat_i = feat.flatten(2).permute(0, 2, 1)
            pos_i = self.build_2d_sincos_position_embedding(
                w, h, self.hidden_dim, device=feat.device
            ).to(dtype=feat.dtype)
            pos_i = pos_i + self.level_embed[i].view(1, 1, -1).to(dtype=feat.dtype)
            feat_flatten.append(feat_i)
            pos_flatten.append(pos_i.repeat(b, 1, 1))

        return torch.cat(feat_flatten, 1), torch.cat(pos_flatten, 1), spatial_shapes

    def _generate_anchors(self, spatial_shapes, dtype=torch.float32, device="cpu"):
        anchors = []
        for lvl, (h, w) in enumerate(spatial_shapes):
            grid_y, grid_x = torch.meshgrid(
                torch.arange(h, device=device), torch.arange(w, device=device), indexing="ij"
            )
            grid_xy = torch.stack([grid_x, grid_y], dim=-1).to(dtype)
            grid_xy = (grid_xy + 0.5) / torch.tensor([w, h], dtype=dtype, device=device)
            wh = torch.ones_like(grid_xy) * 0.05 * (2.0 ** lvl)
            anchors.append(torch.cat([grid_xy, wh], dim=-1).reshape(1, h * w, 4))
        anchors = torch.cat(anchors, dim=1)
        valid_mask = ((anchors > self.eps) & (anchors < 1 - self.eps)).all(-1, keepdim=True)
        anchors_unact = inverse_sigmoid(anchors.clamp(0, 1))
        anchors_unact = torch.where(valid_mask, anchors_unact, torch.inf)
        return anchors_unact, valid_mask

    def _select_topk(self, memory, logits, anchors_unact, topk):
        if self.query_select_method == "agnostic":
            scores = logits.squeeze(-1)
        elif self.query_select_method == "one2many":
            _, topk_ind = torch.topk(logits.flatten(1), topk, dim=-1)
            topk_ind = topk_ind // self.num_classes
            return self._gather_topk(memory, logits, anchors_unact, topk_ind)
        else:
            scores = logits.max(-1).values
        _, topk_ind = torch.topk(scores, topk, dim=-1)
        return self._gather_topk(memory, logits, anchors_unact, topk_ind)

    @staticmethod
    def _gather_topk(memory, logits, anchors_unact, topk_ind):
        topk_memory = memory.gather(1, topk_ind.unsqueeze(-1).repeat(1, 1, memory.shape[-1]))
        topk_logits = logits.gather(1, topk_ind.unsqueeze(-1).repeat(1, 1, logits.shape[-1]))
        topk_anchors = anchors_unact.gather(1, topk_ind.unsqueeze(-1).repeat(1, 1, anchors_unact.shape[-1]))
        return topk_memory, topk_logits, topk_anchors

    def _get_decoder_input(self, memory, spatial_shapes):
        anchors_unact, valid_mask = self._generate_anchors(
            spatial_shapes, dtype=memory.dtype, device=memory.device
        )
        if memory.shape[0] > 1:
            anchors_unact = anchors_unact.repeat(memory.shape[0], 1, 1)
            valid_mask = valid_mask.repeat(memory.shape[0], 1, 1)

        memory = memory * valid_mask.to(memory.dtype)
        output_memory = self.enc_output(memory)
        enc_logits = self.enc_score_head(output_memory)
        topk_memory, topk_logits, topk_anchors = self._select_topk(
            output_memory, enc_logits, anchors_unact, self.num_queries
        )
        enc_topk_bboxes = torch.sigmoid(self.enc_bbox_head(topk_memory) + topk_anchors)

        if self.learn_query_content:
            content = self.tgt_embed.weight.unsqueeze(0).repeat(memory.shape[0], 1, 1)
        else:
            content = topk_memory.detach()
        return content, topk_anchors.detach(), [enc_topk_bboxes], [topk_logits]

    def _make_rbox(self, box4, angle_raw):
        angle = torch.tanh(angle_raw) * (math.pi / 2.0)
        return torch.cat([box4.clamp(0, 1), angle], dim=-1)

    def _format_box(self, box4, angle_raw=None):
        box4 = box4.clamp(0, 1)
        if not self.predict_angle:
            return box4
        if angle_raw is None:
            angle_raw = box4.new_zeros((*box4.shape[:-1], 1))
        return self._make_rbox(box4, angle_raw)

    def forward(self, feats, targets=None):
        memory, memory_pos, spatial_shapes = self._get_encoder_input(feats)
        target, ref_points_unact, enc_topk_bboxes_list, enc_topk_logits_list = self._get_decoder_input(
            memory, spatial_shapes
        )

        output = target
        ref_points = torch.sigmoid(ref_points_unact)
        output_boxes, output_logits = [], []

        pre_box4 = torch.sigmoid(self.pre_bbox_head(output) + inverse_sigmoid(ref_points))
        pre_angle = self.pre_angle_head(output) if self.predict_angle else None
        pre_box = self._format_box(pre_box4, pre_angle)
        pre_logits = self.dec_score_head[0](output)

        for i, layer in enumerate(self.layers):
            query_pos = self.query_pos_head(ref_points).clamp(min=-10, max=10)
            output = layer(output, memory, query_pos=query_pos, memory_pos=memory_pos)

            delta = self.dec_bbox_head[i](output)
            box4 = torch.sigmoid(delta + inverse_sigmoid(ref_points))
            angle = self.dec_angle_head[i](output) if self.predict_angle else None
            det_box = self._format_box(box4, angle)
            logits = self.dec_score_head[i](output)

            output_boxes.append(det_box)
            output_logits.append(logits)
            ref_points = box4.detach()

        out = {
            "pred_logits": output_logits[-1],
            "pred_boxes": output_boxes[-1],
        }

        if self.training and self.aux_loss:
            out["aux_outputs"] = self._set_aux_loss(output_logits[:-1], output_boxes[:-1])
            enc_aux = []
            for a, b in zip(enc_topk_logits_list, enc_topk_bboxes_list):
                enc_aux.append({"pred_logits": a, "pred_boxes": self._format_box(b)})
            out["enc_aux_outputs"] = enc_aux
            out["pre_outputs"] = {"pred_logits": pre_logits, "pred_boxes": pre_box}
            out["enc_meta"] = {"class_agnostic": False}

        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        return [{"pred_logits": a, "pred_boxes": b} for a, b in zip(outputs_class, outputs_coord)]

    def convert_to_deploy(self):
        self.eval()
        return self
