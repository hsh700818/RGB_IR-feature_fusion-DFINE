#!/usr/bin/env python3
"""Profile forward GFLOPs for the RGB-IR D-FINE model.

This script builds the model from the same YAMLConfig path used by train.py,
runs a real forward pass with a 6-channel RGB-IR dummy tensor, and reports
forward GFLOPs/GMACs/parameter counts.

It is designed for the current dual-stream + IACF + GWD-OBB code path:
  RGB-IR tensor -> backbone/backbone_ir -> IACF fusion layers -> encoder -> decoder

Compared with generic FLOPs tools, this script also counts common operations
that are easy to miss in custom fusion modules, including torch.bmm in CMI,
functional softmax, interpolation, adaptive pooling, deformable convolution
when torchvision.ops.DeformConv2d is used, and element-wise costs inside
GLSA/AlignmentAwareFusion/CrossModalInteraction/AdvancedMultimodalFusion.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Importing src triggers the project registry side effects.
import src  # noqa: F401
from src.core import YAMLConfig, yaml_utils


def _numel(x: Any) -> int:
    if torch.is_tensor(x):
        return int(x.numel())
    if isinstance(x, (list, tuple)):
        return sum(_numel(v) for v in x)
    if isinstance(x, dict):
        return sum(_numel(v) for v in x.values())
    return 0


def _shape(x: Any) -> str:
    if torch.is_tensor(x):
        return "x".join(str(v) for v in x.shape)
    if isinstance(x, (list, tuple)) and x:
        return "[" + ", ".join(_shape(v) for v in x[:3]) + (", ..." if len(x) > 3 else "") + "]"
    if isinstance(x, dict):
        return "{" + ", ".join(f"{k}: {_shape(v)}" for k, v in list(x.items())[:3]) + (", ..." if len(x) > 3 else "") + "}"
    return type(x).__name__


def _to_2tuple(v: Any) -> Tuple[int, int]:
    if isinstance(v, tuple):
        return int(v[0]), int(v[1])
    return int(v), int(v)


def _get_model_state_dict(checkpoint: Any):
    if isinstance(checkpoint, dict):
        if "ema" in checkpoint and isinstance(checkpoint["ema"], dict) and "module" in checkpoint["ema"]:
            return checkpoint["ema"]["module"]
        if "model" in checkpoint and isinstance(checkpoint["model"], dict):
            return checkpoint["model"]
    return checkpoint


def _load_checkpoint_if_needed(model: nn.Module, path: str | None, strict: bool = False) -> None:
    if not path:
        return
    if path.startswith("http"):
        raise ValueError("profile_gflops.py only supports local checkpoint paths. Please download the weight file first.")
    ckpt_path = Path(path)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    checkpoint = torch.load(str(ckpt_path), map_location="cpu")
    state = _get_model_state_dict(checkpoint)
    if not isinstance(state, dict):
        raise TypeError(f"Unsupported checkpoint format: {type(checkpoint)}")

    model_state = model.state_dict()
    filtered = {}
    skipped = []
    for k, v in state.items():
        if k in model_state and tuple(model_state[k].shape) == tuple(v.shape):
            filtered[k] = v
        else:
            skipped.append(k)

    missing, unexpected = model.load_state_dict(filtered, strict=False)
    print(f"Loaded checkpoint: {ckpt_path}")
    print(f"  matched tensors: {len(filtered)}")
    print(f"  skipped tensors: {len(skipped)}")
    if strict and (missing or unexpected or skipped):
        raise RuntimeError(
            f"Strict loading failed. missing={len(missing)}, unexpected={len(unexpected)}, skipped={len(skipped)}"
        )


def build_model(args: argparse.Namespace) -> nn.Module:
    update_dict = yaml_utils.parse_cli(args.update) if args.update else {}
    update_dict.update({"device": args.device})

    cfg = YAMLConfig(args.config, **update_dict)

    # FLOPs do not depend on pretrained weights. Disable HGNetv2 pretrained
    # download to make profiling work on servers without internet access.
    if "HGNetv2" in cfg.yaml_cfg:
        cfg.yaml_cfg["HGNetv2"]["pretrained"] = False

    model = cfg.model
    _load_checkpoint_if_needed(model, args.resume or args.tuning, strict=args.strict_load)
    model.to(args.device)
    model.eval()
    return model


class ForwardFlopsCounter:
    """Runtime FLOPs counter based on module hooks plus selected function patches."""

    def __init__(self, model: nn.Module):
        self.model = model
        self.total_flops = 0
        self.by_module: Dict[str, int] = defaultdict(int)
        self.handles: List[Any] = []
        self.name_of: Dict[nn.Module, str] = {m: n for n, m in model.named_modules()}
        self._orig: Dict[str, Any] = {}

    def add(self, key: str, flops: int | float) -> None:
        flops = int(flops)
        if flops <= 0:
            return
        self.total_flops += flops
        self.by_module[key] += flops

    def _module_name(self, module: nn.Module, fallback: str) -> str:
        return self.name_of.get(module, fallback) or fallback

    def _conv_hook(self, module: nn.Module, inputs: Tuple[Any, ...], output: Any) -> None:
        if not torch.is_tensor(output):
            return
        x = inputs[0]
        if not torch.is_tensor(x):
            return
        batch = output.shape[0]
        out_c = output.shape[1]
        out_h = output.shape[2] if output.dim() > 2 else 1
        out_w = output.shape[3] if output.dim() > 3 else 1
        kernel_h, kernel_w = _to_2tuple(module.kernel_size)
        groups = int(getattr(module, "groups", 1))
        in_c = int(x.shape[1])
        conv_per_position = kernel_h * kernel_w * in_c // groups
        # one multiply and one add per MAC, which matches GFLOPs = 2 * GMACs.
        flops = batch * out_c * out_h * out_w * conv_per_position * 2
        if getattr(module, "bias", None) is not None:
            flops += batch * out_c * out_h * out_w
        self.add(self._module_name(module, module.__class__.__name__), flops)

    def _linear_hook(self, module: nn.Linear, inputs: Tuple[Any, ...], output: Any) -> None:
        if not torch.is_tensor(inputs[0]) or not torch.is_tensor(output):
            return
        out_numel = output.numel()
        flops = out_numel * int(module.in_features) * 2
        if module.bias is not None:
            flops += out_numel
        self.add(self._module_name(module, "Linear"), flops)

    def _norm_hook(self, module: nn.Module, inputs: Tuple[Any, ...], output: Any) -> None:
        # Approximate affine normalization cost. Statistics are small compared with conv/mm.
        self.add(self._module_name(module, module.__class__.__name__), _numel(output) * 2)

    def _activation_hook(self, module: nn.Module, inputs: Tuple[Any, ...], output: Any) -> None:
        self.add(self._module_name(module, module.__class__.__name__), _numel(output))

    def _pool_hook(self, module: nn.Module, inputs: Tuple[Any, ...], output: Any) -> None:
        if not torch.is_tensor(inputs[0]):
            return
        x = inputs[0]
        out_numel = _numel(output)
        if isinstance(module, (nn.AvgPool2d, nn.MaxPool2d)):
            kh, kw = _to_2tuple(module.kernel_size)
            self.add(self._module_name(module, module.__class__.__name__), out_numel * kh * kw)
        else:
            # Adaptive pooling has data-dependent regions. Use input/output ratio.
            ratio = max(int(x.numel() / max(out_numel, 1)), 1)
            self.add(self._module_name(module, module.__class__.__name__), out_numel * ratio)

    def _upsample_hook(self, module: nn.Upsample, inputs: Tuple[Any, ...], output: Any) -> None:
        # Bilinear ~= several multiplies/adds per output; nearest ~= one copy.
        mode = getattr(module, "mode", "nearest")
        factor = 8 if mode in {"linear", "bilinear", "bicubic", "trilinear"} else 1
        self.add(self._module_name(module, "Upsample"), _numel(output) * factor)

    def _special_missing_elementwise_hook(self, module: nn.Module, inputs: Tuple[Any, ...], output: Any) -> None:
        name = module.__class__.__name__
        key = self._module_name(module, name) + ":elementwise_extra"

        if name == "GLSA" and torch.is_tensor(output):
            # x_h.sigmoid, x_w.sigmoid, and three broadcast multiplications.
            b, c, h, w = output.shape
            flops = b * c * h + b * c * w + 3 * output.numel()
            self.add(key, flops)
        elif name == "AlignmentAwareFusion" and torch.is_tensor(output):
            # residual alignment scale/add and gated weighted sum.
            self.add(key, output.numel() * 5)
        elif name == "CrossModalInteraction" and torch.is_tensor(output):
            # gamma scaling and residual add after bmm/interpolate are counted separately.
            self.add(key, output.numel() * 2)
        elif name == "AdvancedMultimodalFusion" and torch.is_tensor(output):
            # illumination/gate reweighting and two feature-map multiplications before fusion_conv.
            self.add(key, output.numel() * 4)

    def _patch_functions(self):
        counter = self

        self._orig["torch.bmm"] = torch.bmm
        def bmm_hook(a, b, *args, **kwargs):
            out = counter._orig["torch.bmm"](a, b, *args, **kwargs)
            if torch.is_tensor(a) and torch.is_tensor(b):
                batch, n, k = a.shape
                m = b.shape[-1]
                counter.add("torch.bmm", batch * n * m * k * 2)
            return out
        torch.bmm = bmm_hook

        self._orig["torch.matmul"] = torch.matmul
        def matmul_hook(a, b, *args, **kwargs):
            out = counter._orig["torch.matmul"](a, b, *args, **kwargs)
            if torch.is_tensor(a) and torch.is_tensor(b) and torch.is_tensor(out):
                if a.dim() >= 2 and b.dim() >= 2:
                    k = a.shape[-1]
                    counter.add("torch.matmul", out.numel() * k * 2)
            return out
        torch.matmul = matmul_hook

        self._orig["F.softmax"] = F.softmax
        def softmax_hook(input, *args, **kwargs):
            out = counter._orig["F.softmax"](input, *args, **kwargs)
            # exp + sum + div, rough but useful for attention modules.
            counter.add("functional.softmax", _numel(out) * 3)
            return out
        F.softmax = softmax_hook

        self._orig["torch.softmax"] = torch.softmax
        def torch_softmax_hook(input, *args, **kwargs):
            out = counter._orig["torch.softmax"](input, *args, **kwargs)
            counter.add("torch.softmax", _numel(out) * 3)
            return out
        torch.softmax = torch_softmax_hook

        self._orig["F.interpolate"] = F.interpolate
        def interpolate_hook(input, *args, **kwargs):
            out = counter._orig["F.interpolate"](input, *args, **kwargs)
            mode = kwargs.get("mode", None)
            if mode is None and len(args) >= 3:
                mode = args[2]
            factor = 8 if mode in {"linear", "bilinear", "bicubic", "trilinear"} else 1
            counter.add("functional.interpolate", _numel(out) * factor)
            return out
        F.interpolate = interpolate_hook

    def _restore_functions(self):
        if "torch.bmm" in self._orig:
            torch.bmm = self._orig["torch.bmm"]
        if "torch.matmul" in self._orig:
            torch.matmul = self._orig["torch.matmul"]
        if "F.softmax" in self._orig:
            F.softmax = self._orig["F.softmax"]
        if "torch.softmax" in self._orig:
            torch.softmax = self._orig["torch.softmax"]
        if "F.interpolate" in self._orig:
            F.interpolate = self._orig["F.interpolate"]

    def __enter__(self):
        for module in self.model.modules():
            class_name = module.__class__.__name__
            if isinstance(module, nn.Conv2d) or class_name == "DeformConv2d":
                self.handles.append(module.register_forward_hook(self._conv_hook))
            elif isinstance(module, nn.Linear):
                self.handles.append(module.register_forward_hook(self._linear_hook))
            elif isinstance(module, (nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)):
                self.handles.append(module.register_forward_hook(self._norm_hook))
            elif isinstance(module, (nn.ReLU, nn.ReLU6, nn.SiLU, nn.GELU, nn.Sigmoid, nn.Tanh, nn.Softmax)):
                self.handles.append(module.register_forward_hook(self._activation_hook))
            elif isinstance(module, (nn.AdaptiveAvgPool2d, nn.AvgPool2d, nn.MaxPool2d)):
                self.handles.append(module.register_forward_hook(self._pool_hook))
            elif isinstance(module, nn.Upsample):
                self.handles.append(module.register_forward_hook(self._upsample_hook))

            if class_name in {
                "GLSA",
                "AlignmentAwareFusion",
                "CrossModalInteraction",
                "AdvancedMultimodalFusion",
            }:
                self.handles.append(module.register_forward_hook(self._special_missing_elementwise_hook))

        self._patch_functions()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        for h in self.handles:
            h.remove()
        self._restore_functions()

    def top_modules(self, n: int = 30) -> List[Tuple[str, int]]:
        return sorted(self.by_module.items(), key=lambda kv: kv[1], reverse=True)[:n]


def count_params(model: nn.Module) -> Tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile forward GFLOPs of the RGB-IR D-FINE model.")
    parser.add_argument("-c", "--config", required=True, help="Path to model YAML config.")
    parser.add_argument("--input-size", nargs=2, type=int, default=[640, 640], metavar=("H", "W"))
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("-d", "--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("-r", "--resume", default=None, help="Optional checkpoint to load before profiling.")
    parser.add_argument("-t", "--tuning", default=None, help="Optional tuning checkpoint to load before profiling.")
    parser.add_argument("--strict-load", action="store_true", help="Fail if checkpoint tensors do not fully match.")
    parser.add_argument("--amp", action="store_true", help="Run the dummy forward under autocast on CUDA.")
    parser.add_argument("--topk", type=int, default=30, help="Print top-k modules by FLOPs.")
    parser.add_argument("-u", "--update", nargs="+", help="Override YAML fields, same style as train.py -u.")
    args = parser.parse_args()

    if args.resume and args.tuning:
        raise ValueError("Use only one of --resume or --tuning.")

    model = build_model(args)
    h, w = args.input_size
    dummy = torch.randn(args.batch_size, 6, h, w, device=args.device)

    autocast_ctx = torch.cuda.amp.autocast if args.amp and str(args.device).startswith("cuda") else contextlib.nullcontext

    with torch.no_grad():
        with ForwardFlopsCounter(model) as counter:
            with autocast_ctx():
                output = model(dummy)

    total_params, trainable_params = count_params(model)
    flops = counter.total_flops
    macs = flops / 2.0

    print("\n================ RGB-IR D-FINE Forward Complexity ================")
    print(f"Config:              {args.config}")
    print(f"Input tensor:         {args.batch_size} x 6 x {h} x {w}")
    print(f"Output structure:     {_shape(output)}")
    print(f"Total params:         {total_params / 1e6:.4f} M")
    print(f"Trainable params:     {trainable_params / 1e6:.4f} M")
    print(f"Forward MACs:         {macs / 1e9:.4f} GMACs")
    print(f"Forward FLOPs:        {flops / 1e9:.4f} GFLOPs")
    print("Note: FLOPs are reported as 2 x MACs for conv/linear/matmul operations.")
    print("      Training forward+backward cost is commonly about 3x forward FLOPs.\n")

    print(f"Top {args.topk} FLOPs entries:")
    for name, value in counter.top_modules(args.topk):
        print(f"  {name:<70s} {value / 1e9:>10.4f} GFLOPs")


if __name__ == "__main__":
    main()
