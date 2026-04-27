import math
import sys
from typing import Dict, Iterable, List

import numpy as np
import torch
import torch.amp
from torch.cuda.amp.grad_scaler import GradScaler
from torch.utils.tensorboard import SummaryWriter

from ..data import CocoEvaluator
from ..data.dataset import mscoco_category2label, mscoco_category2name
from ..misc import MetricLogger, SmoothedValue, dist_utils, save_samples
from ..optim import ModelEMA, Warmup
from .validator import Validator, scale_boxes


def train_one_epoch(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    use_wandb: bool,
    max_norm: float = 0,
    **kwargs,
):
    if use_wandb:
        import wandb

    model.train()
    criterion.train()
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))

    epochs = kwargs.get("epochs", None)
    header = "Epoch: [{}]".format(epoch) if epochs is None else "Epoch: [{}/{}]".format(epoch, epochs)

    print_freq = kwargs.get("print_freq", 10)
    writer: SummaryWriter = kwargs.get("writer", None)

    ema: ModelEMA = kwargs.get("ema", None)
    scaler: GradScaler = kwargs.get("scaler", None)
    lr_warmup_scheduler: Warmup = kwargs.get("lr_warmup_scheduler", None)
    losses = []

    output_dir = kwargs.get("output_dir", None)
    num_visualization_sample_batch = kwargs.get("num_visualization_sample_batch", 1)

    for i, (samples, targets) in enumerate(
        metric_logger.log_every(data_loader, print_freq, header)
    ):
        global_step = epoch * len(data_loader) + i
        metas = dict(epoch=epoch, step=i, global_step=global_step, epoch_step=len(data_loader))

        if global_step < num_visualization_sample_batch and output_dir is not None and dist_utils.is_main_process():
            save_samples(samples, targets, output_dir, "train", normalized=True, box_fmt="cxcywh")

        samples = samples.to(device)
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]

        if scaler is not None:
            with torch.autocast(device_type=str(device), cache_enabled=True):
                outputs = model(samples, targets=targets)

            if torch.isnan(outputs["pred_boxes"]).any() or torch.isinf(outputs["pred_boxes"]).any():
                print(outputs["pred_boxes"])
                state = model.state_dict()
                new_state = {}
                for key, value in model.state_dict().items():
                    new_key = key.replace("module.", "")
                    state[new_key] = value
                new_state["model"] = state
                dist_utils.save_on_master(new_state, "./NaN.pth")

            with torch.autocast(device_type=str(device), enabled=False):
                loss_dict = criterion(outputs, targets, **metas)

            loss = sum(loss_dict.values())
            scaler.scale(loss).backward()

            if max_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        else:
            outputs = model(samples, targets=targets)
            loss_dict = criterion(outputs, targets, **metas)

            loss: torch.Tensor = sum(loss_dict.values())
            optimizer.zero_grad()
            loss.backward()

            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

            optimizer.step()

        if ema is not None:
            ema.update(model)

        if lr_warmup_scheduler is not None:
            lr_warmup_scheduler.step()

        loss_dict_reduced = dist_utils.reduce_dict(loss_dict)
        loss_value = sum(loss_dict_reduced.values())
        losses.append(loss_value.detach().cpu().numpy())

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        metric_logger.update(loss=loss_value, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        if writer and dist_utils.is_main_process() and global_step % 10 == 0:
            writer.add_scalar("Loss/total", loss_value.item(), global_step)
            for j, pg in enumerate(optimizer.param_groups):
                writer.add_scalar(f"Lr/pg_{j}", pg["lr"], global_step)
            for k, v in loss_dict_reduced.items():
                writer.add_scalar(f"Loss/{k}", v.item(), global_step)

    if use_wandb:
        wandb.log(
            {"lr": optimizer.param_groups[0]["lr"], "epoch": epoch, "train/loss": np.mean(losses)}
        )
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def _compute_coco_ap50_per_class(coco_eval, category2name=None):
    """Return overall mAP@0.5 and per-class AP@0.5 from a COCO-style evaluator.

    COCO precision shape is [T, R, K, A, M]: IoU thresholds, recall thresholds,
    categories, area ranges, and max detections.  AP@0.5 uses IoU=0.50, area=all,
    and maxDets=100 when available.
    """
    if coco_eval is None or not hasattr(coco_eval, "eval") or coco_eval.eval is None:
        return {}, None

    precision = coco_eval.eval.get("precision", None)
    if precision is None:
        return {}, None

    params = coco_eval.params
    iou_thrs = np.asarray(params.iouThrs)
    cat_ids = list(params.catIds)
    area_lbl = list(getattr(params, "areaRngLbl", ["all"]))
    max_dets = list(getattr(params, "maxDets", [100]))

    iou_idx = int(np.argmin(np.abs(iou_thrs - 0.5)))
    area_idx = area_lbl.index("all") if "all" in area_lbl else 0
    maxdet_idx = max_dets.index(100) if 100 in max_dets else len(max_dets) - 1

    per_class_ap50 = {}
    ap_values = []
    for k, cat_id in enumerate(cat_ids):
        precision_k = precision[iou_idx, :, k, area_idx, maxdet_idx]
        precision_k = precision_k[precision_k > -1]
        ap = float(np.mean(precision_k)) if precision_k.size else float("nan")
        name = category2name.get(cat_id, str(cat_id)) if category2name else str(cat_id)
        per_class_ap50[name] = ap
        if not math.isnan(ap):
            ap_values.append(ap)

    map50 = float(np.mean(ap_values)) if ap_values else float("nan")
    return per_class_ap50, map50


def _print_ap50_table(per_class_ap50, map50):
    if not per_class_ap50:
        return
    print("\nDroneVehicle AP@0.5 per class:")
    print("+-------------+--------+")
    print("| class       | AP@0.5 |")
    print("+-------------+--------+")
    for name, ap in per_class_ap50.items():
        ap_str = "nan" if math.isnan(ap) else f"{ap:.4f}"
        print(f"| {name:<11} | {ap_str:>6} |")
    map_str = "nan" if math.isnan(map50) else f"{map50:.4f}"
    print("+-------------+--------+")
    print(f"| {'mAP@0.5':<11} | {map_str:>6} |")
    print("+-------------+--------+\n")


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    postprocessor,
    data_loader,
    coco_evaluator: CocoEvaluator,
    device,
    epoch: int,
    use_wandb: bool,
    **kwargs,
):
    if use_wandb:
        import wandb

    model.eval()
    criterion.eval()
    coco_evaluator.cleanup()

    metric_logger = MetricLogger(delimiter="  ")
    header = "Test:"

    iou_types = coco_evaluator.iou_types

    gt: List[Dict[str, torch.Tensor]] = []
    preds: List[Dict[str, torch.Tensor]] = []

    output_dir = kwargs.get("output_dir", None)
    num_visualization_sample_batch = kwargs.get("num_visualization_sample_batch", 1)

    for i, (samples, targets) in enumerate(metric_logger.log_every(data_loader, 10, header)):
        global_step = epoch * len(data_loader) + i

        if i < num_visualization_sample_batch and output_dir is not None and dist_utils.is_main_process():
            save_samples(samples, targets, output_dir, "val", normalized=False, box_fmt="xyxy")

        samples = samples.to(device)
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]

        outputs = model(samples)

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessor(outputs, orig_target_sizes)

        res = {target["image_id"].item(): output for target, output in zip(targets, results)}
        if coco_evaluator is not None:
            coco_evaluator.update(res)

        for idx, (target, result) in enumerate(zip(targets, results)):
            gt.append(
                {
                    "boxes": scale_boxes(
                        target["boxes"],
                        (target["orig_size"][1], target["orig_size"][0]),
                        (samples[idx].shape[-1], samples[idx].shape[-2]),
                    ),
                    "labels": target["labels"],
                }
            )
            labels = (
                torch.tensor([mscoco_category2label[int(x.item())] for x in result["labels"].flatten()])
                .to(result["labels"].device)
                .reshape(result["labels"].shape)
            ) if postprocessor.remap_mscoco_category else result["labels"]
            preds.append(
                {"boxes": result["boxes"], "labels": labels, "scores": result["scores"]}
            )

    metrics = Validator(gt, preds).compute_metrics()
    print("Metrics:", metrics)
    if use_wandb:
        metrics = {f"metrics/{k}": v for k, v in metrics.items()}
        metrics["epoch"] = epoch
        wandb.log(metrics)

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()

    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()

    stats = {}
    if coco_evaluator is not None:
        if "bbox" in iou_types:
            bbox_eval = coco_evaluator.coco_eval["bbox"]
            stats["coco_eval_bbox"] = bbox_eval.stats.tolist()
            per_class_ap50, map50 = _compute_coco_ap50_per_class(bbox_eval, mscoco_category2name)
            _print_ap50_table(per_class_ap50, map50)
            stats["mAP50"] = map50
            for name, ap in per_class_ap50.items():
                stats[f"AP50_{name}"] = ap
        if "segm" in iou_types:
            stats["coco_eval_masks"] = coco_evaluator.coco_eval["segm"].stats.tolist()

    return stats, coco_evaluator
