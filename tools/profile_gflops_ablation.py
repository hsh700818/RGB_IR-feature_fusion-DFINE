#!/usr/bin/env python3
"""Profile forward GFLOPs for RGB-IR D-FINE with optional ablation of IACF or GWD-OBB modules."""

from __future__ import annotations

import argparse
import contextlib
import torch
from pathlib import Path
from collections import defaultdict
from src.core import YAMLConfig, yaml_utils
from src.zoo.dfine.dfine import DFINE

# Reuse ForwardFlopsCounter from profile_gflops.py
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[0]))
from profile_gflops import ForwardFlopsCounter, count_params, _shape

def build_model(args: argparse.Namespace) -> torch.nn.Module:
    update_dict = yaml_utils.parse_cli(args.update) if args.update else {}
    update_dict.update({"device": args.device})

    cfg = YAMLConfig(args.config, **update_dict)

    if "HGNetv2" in cfg.yaml_cfg:
        cfg.yaml_cfg["HGNetv2"]["pretrained"] = False

    model = cfg.model

    # Apply ablation flags
    for module in model.modules():
        if hasattr(module, 'use_cmi') and args.no_iacf:
            module.use_cmi = False
        if hasattr(module, 'use_glsa') and args.no_gwd_obb:
            module.use_glsa = False

    model.to(args.device)
    model.eval()
    return model

def main():
    parser = argparse.ArgumentParser(description="Profile GFLOPs with IACF/GWD-OBB ablation")
    parser.add_argument('-c', '--config', required=True, help='YAML config path')
    parser.add_argument('--input-size', nargs=2, type=int, default=[640, 640], help='Input H W')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--no-iacf', action='store_true', help='Disable IACF module')
    parser.add_argument('--no-gwd-obb', action='store_true', help='Disable GWD-OBB Head (GLSA)')
    parser.add_argument('-u', '--update', nargs='+', help='Override YAML fields')
    args = parser.parse_args()

    model = build_model(args)

    h, w = args.input_size
    dummy = torch.randn(args.batch_size, 6, h, w, device=args.device)

    autocast_ctx = contextlib.nullcontext
    with torch.no_grad():
        with ForwardFlopsCounter(model) as counter:
            with autocast_ctx():
                output = model(dummy)

    total, trainable = count_params(model)
    flops = counter.total_flops
    macs = flops / 2.0

    print("\n================ RGB-IR D-FINE Ablation GFLOPs ================")
    print(f"Input tensor: {args.batch_size} x 6 x {h} x {w}")
    print(f"Output structure: {_shape(output)}")
    print(f"Total params: {total / 1e6:.4f} M")
    print(f"Trainable params: {trainable / 1e6:.4f} M")
    print(f"Forward MACs: {macs / 1e9:.4f} GMACs")
    print(f"Forward FLOPs: {flops / 1e9:.4f} GFLOPs")
    print("Note: FLOPs are reported as 2 x MACs for conv/linear/matmul operations.")

    print("Top 30 FLOPs entries:")
    for name, value in counter.top_modules(30):
        print(f"  {name:<70s} {value / 1e9:>10.4f} GFLOPs")

if __name__ == '__main__':
    main()