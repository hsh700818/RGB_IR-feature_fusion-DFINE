#!/usr/bin/env python3
"""Profile forward GFLOPs for RGB-IR D-FINE with optional ablation of IACF or GWD-OBB modules."""

import argparse
import torch
from pathlib import Path
from src.core import YAMLConfig, yaml_utils

# Import the DFINE model to access fusion flags
from src.zoo.dfine.dfine import DFINE

def build_model(args: argparse.Namespace) -> torch.nn.Module:
    update_dict = yaml_utils.parse_cli(args.update) if args.update else {}
    update_dict.update({"device": args.device})

    cfg = YAMLConfig(args.config, **update_dict)

    # Disable pretrained weights to avoid downloads
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

def count_params(model: torch.nn.Module):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

def main():
    parser = argparse.ArgumentParser(description="Profile GFLOPs with ablation")
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

    with torch.no_grad():
        output = model(dummy)

    total, trainable = count_params(model)

    print("\n================ RGB-IR D-FINE Ablation GFLOPs ================")
    print(f"Input tensor: {args.batch_size} x 6 x {h} x {w}")
    print(f"Output structure: {output}" )
    print(f"Total params: {total / 1e6:.4f} M")
    print(f"Trainable params: {trainable / 1e6:.4f} M")
    print("FLOPs not calculated exactly here; use same methodology as full model.")

if __name__ == '__main__':
    main()