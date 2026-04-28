import argparse
import copy
import os
import sys
from pathlib import Path
from pprint import pprint

import torch

# Ensure that `src` can be imported no matter where the script is launched from.
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core import YAMLConfig, yaml_utils
from src.misc import dist_utils
from src.solver import TASKS


def safe_get_rank():
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    return 0


def _get_model_state_dict(checkpoint):
    """Return the model state dict inside common D-FINE checkpoint formats."""
    if isinstance(checkpoint, dict):
        if "ema" in checkpoint and isinstance(checkpoint["ema"], dict) and "module" in checkpoint["ema"]:
            return checkpoint["ema"]["module"], ("ema", "module")
        if "model" in checkpoint and isinstance(checkpoint["model"], dict):
            return checkpoint["model"], ("model",)
    return checkpoint, ()


def _set_model_state_dict(checkpoint, state_dict, key_path):
    if key_path == ("ema", "module"):
        checkpoint["ema"]["module"] = state_dict
    elif key_path == ("model",):
        checkpoint["model"] = state_dict
    else:
        checkpoint = state_dict
    return checkpoint


def _prepare_dual_stream_tuning_checkpoint(tuning_path, output_dir):
    """
    Expand a single-stream D-FINE checkpoint for the RGB-IR dual-stream model.

    Original COCO checkpoints usually contain `backbone.*` only. Our model has
    both `backbone.*` and `backbone_ir.*`. This helper mirrors backbone weights
    to `backbone_ir.*` before BaseSolver.load_tuning_state performs shape-based
    matching, so the IR branch is not randomly initialized when using `-t`.
    """
    if tuning_path is None:
        return None
    if tuning_path.startswith("http"):
        # Avoid downloading remote checkpoints twice. Remote checkpoints can
        # still be used, but backbone_ir will rely on the model initializer.
        return tuning_path

    src_path = Path(tuning_path)
    if not src_path.is_file():
        return tuning_path

    output_dir = Path(output_dir)
    cache_dir = output_dir / ".cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    expanded_path = cache_dir / f"dual_stream_{src_path.name}"

    is_main = safe_get_rank() == 0
    if is_main:
        checkpoint = torch.load(str(src_path), map_location="cpu")
        state_dict, key_path = _get_model_state_dict(checkpoint)

        if isinstance(state_dict, dict):
            has_rgb_backbone = any(k.startswith("backbone.") for k in state_dict.keys())
            has_ir_backbone = any(k.startswith("backbone_ir.") for k in state_dict.keys())

            if has_rgb_backbone and not has_ir_backbone:
                expanded_state = copy.copy(state_dict)
                for k, v in state_dict.items():
                    if k.startswith("backbone."):
                        expanded_state[k.replace("backbone.", "backbone_ir.", 1)] = v
                checkpoint = _set_model_state_dict(checkpoint, expanded_state, key_path)
                torch.save(checkpoint, str(expanded_path))
                print(f"已为双流模型生成 tuning 权重: {expanded_path}")
            else:
                expanded_path = src_path
                print(f"tuning 权重无需双流扩展: {src_path}")
        else:
            expanded_path = src_path

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()

    return str(expanded_path)


def main(args) -> None:
    dist_utils.setup_distributed(args.print_rank, args.print_method, seed=args.seed)

    assert not all(
        [args.tuning, args.resume]
    ), "Only support from_scratch or resume or tuning at one time"

    update_dict = yaml_utils.parse_cli(args.update)
    update_dict.update(
        {
            k: v
            for k, v in args.__dict__.items()
            if k not in ["update", "config", "local_rank"] and v is not None
        }
    )

    cfg = YAMLConfig(args.config, **update_dict)

    # When loading a full checkpoint, do not also initialize HGNetv2 from its
    # classification pretrained weights. The checkpoint is the source of truth.
    if args.resume or args.tuning:
        if "HGNetv2" in cfg.yaml_cfg:
            cfg.yaml_cfg["HGNetv2"]["pretrained"] = False

    # For RGB-IR dual-stream training, mirror single-stream COCO backbone
    # weights to backbone_ir before BaseSolver loads the tuning state.
    if args.tuning:
        expanded_tuning = _prepare_dual_stream_tuning_checkpoint(
            args.tuning,
            cfg.yaml_cfg.get("output_dir", "./output"),
        )
        cfg.tuning = expanded_tuning
        cfg.yaml_cfg["tuning"] = expanded_tuning

    if safe_get_rank() == 0:
        print("cfg: ")
        pprint(cfg.__dict__)

    solver = TASKS[cfg.yaml_cfg["task"]](cfg)

    if args.test_only:
        solver.val()
    else:
        solver.fit()

    dist_utils.cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-c", "--config", type=str, required=True)
    parser.add_argument("-r", "--resume", type=str, help="resume from checkpoint")
    parser.add_argument("-t", "--tuning", type=str, help="tuning from checkpoint")
    parser.add_argument("-d", "--device", type=str, help="device")
    parser.add_argument("--seed", type=int, help="exp reproducibility")
    parser.add_argument("--use-amp", action="store_true", help="auto mixed precision training")
    parser.add_argument("--output-dir", type=str, help="output directory")
    parser.add_argument("--summary-dir", type=str, help="tensorboard summary")
    parser.add_argument("--test-only", action="store_true", default=False)

    parser.add_argument("-u", "--update", nargs="+", help="update yaml config")

    parser.add_argument("--print-method", type=str, default="builtin", help="print method")
    parser.add_argument("--print-rank", type=int, default=0, help="print rank id")
    parser.add_argument("--local-rank", "--local_rank", dest="local_rank", type=int, help="local rank id")

    args = parser.parse_args()
    main(args)
