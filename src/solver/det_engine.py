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

    for i, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
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
        wandb.log({"lr": optimizer.param_groups[0]["lr"], "epoch": epoch, "train/loss": np.mean(losses)})
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def _compute_coco_ap50_per_class(coco_eval, category2name=None):
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


def _print_ap50_table(per_class_ap50, map50, title="DroneVehicle AP@0.5 per class"):
    if not per_class_ap50:
        return
    print(f"\n{title}:")
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


def _rbox_to_poly_np(rbox):
    cx, cy, w, h, angle = [float(x) for x in rbox]
    w = max(w, 0.0)
    h = max(h, 0.0)
    ca, sa = math.cos(angle), math.sin(angle)
    dx = np.array([-w / 2, w / 2, w / 2, -w / 2], dtype=np.float64)
    dy = np.array([-h / 2, -h / 2, h / 2, h / 2], dtype=np.float64)
    x = cx + dx * ca - dy * sa
    y = cy + dx * sa + dy * ca
    return np.stack([x, y], axis=1)


def _poly_area_np(poly):
    if poly is None or len(poly) < 3:
        return 0.0
    x = poly[:, 0]
    y = poly[:, 1]
    return float(abs(0.5 * (np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))))


def _line_intersection_np(p1, p2, q1, q2):
    r = p2 - p1
    s = q2 - q1
    denom = r[0] * s[1] - r[1] * s[0]
    if abs(denom) < 1e-9:
        return p2
    t = ((q1 - p1)[0] * s[1] - (q1 - p1)[1] * s[0]) / denom
    return p1 + t * r


def _inside_edge_np(p, a, b):
    return ((b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])) >= -1e-9


def _ensure_ccw_np(poly):
    signed = 0.5 * (np.dot(poly[:, 0], np.roll(poly[:, 1], -1)) - np.dot(poly[:, 1], np.roll(poly[:, 0], -1)))
    return poly if signed >= 0 else poly[::-1].copy()


def _convex_clip_np(subject, clip):
    output = _ensure_ccw_np(subject)
    clip = _ensure_ccw_np(clip)
    for i in range(len(clip)):
        a = clip[i]
        b = clip[(i + 1) % len(clip)]
        input_list = output
        if len(input_list) == 0:
            break
        output_pts = []
        s = input_list[-1]
        for e in input_list:
            if _inside_edge_np(e, a, b):
                if not _inside_edge_np(s, a, b):
                    output_pts.append(_line_intersection_np(s, e, a, b))
                output_pts.append(e)
            elif _inside_edge_np(s, a, b):
                output_pts.append(_line_intersection_np(s, e, a, b))
            s = e
        output = np.asarray(output_pts, dtype=np.float64)
    return output


def _rotated_iou_np(box1, box2):
    poly1 = _rbox_to_poly_np(box1)
    poly2 = _rbox_to_poly_np(box2)
    area1 = _poly_area_np(poly1)
    area2 = _poly_area_np(poly2)
    if area1 <= 0 or area2 <= 0:
        return 0.0
    inter_poly = _convex_clip_np(poly1, poly2)
    inter = _poly_area_np(inter_poly)
    union = area1 + area2 - inter
    return 0.0 if union <= 0 else float(inter / union)


def _voc_ap(rec, prec):
    mrec = np.concatenate(([0.0], rec, [1.0]))
    mpre = np.concatenate(([0.0], prec, [0.0]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    i = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1]))


def _compute_obb_ap50(gt, preds, category2name=None, iou_thr=0.5):
    category2name = category2name or {}
    all_classes = sorted(set([int(x) for g in gt for x in g["labels"]]) | set([int(x) for p in preds for x in p["labels"]]))
    per_class_ap = {}
    valid_aps = []

    for cls in all_classes:
        gt_by_image = {}
        npos = 0
        for img_id, g in enumerate(gt):
            labels = g["labels"]
            boxes = g["rboxes"]
            inds = np.where(labels == cls)[0]
            gt_by_image[img_id] = {"boxes": boxes[inds], "detected": np.zeros(len(inds), dtype=bool)}
            npos += len(inds)

        cls_preds = []
        for img_id, p in enumerate(preds):
            labels = p["labels"]
            inds = np.where(labels == cls)[0]
            for ind in inds:
                cls_preds.append((float(p["scores"][ind]), img_id, p["rboxes"][ind]))
        cls_preds.sort(key=lambda x: x[0], reverse=True)

        tp = np.zeros(len(cls_preds), dtype=np.float64)
        fp = np.zeros(len(cls_preds), dtype=np.float64)
        for i, (_, img_id, pred_box) in enumerate(cls_preds):
            gt_info = gt_by_image.get(img_id, {"boxes": np.zeros((0, 5)), "detected": np.zeros(0, dtype=bool)})
            gt_boxes = gt_info["boxes"]
            if len(gt_boxes) == 0:
                fp[i] = 1.0
                continue
            ious = np.array([_rotated_iou_np(pred_box, gt_box) for gt_box in gt_boxes], dtype=np.float64)
            j = int(np.argmax(ious))
            if ious[j] >= iou_thr and not gt_info["detected"][j]:
                tp[i] = 1.0
                gt_info["detected"][j] = True
            else:
                fp[i] = 1.0

        if npos == 0:
            ap = float("nan")
        else:
            fp_cum = np.cumsum(fp)
            tp_cum = np.cumsum(tp)
            rec = tp_cum / max(float(npos), np.finfo(np.float64).eps)
            prec = tp_cum / np.maximum(tp_cum + fp_cum, np.finfo(np.float64).eps)
            ap = _voc_ap(rec, prec)
            valid_aps.append(ap)

        name = category2name.get(cls + 1, str(cls)) if cls < len(category2name) else category2name.get(cls, str(cls))
        per_class_ap[name] = ap

    map50 = float(np.mean(valid_aps)) if valid_aps else float("nan")
    return per_class_ap, map50


def _scale_target_rboxes(target_boxes, orig_size):
    boxes = target_boxes.detach().cpu().numpy().astype(np.float64)
    scale = np.array([float(orig_size[0]), float(orig_size[1]), float(orig_size[0]), float(orig_size[1]), 1.0], dtype=np.float64)
    return boxes * scale[None]


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

    eval_hbb = bool(kwargs.get("eval_hbb", True))
    eval_obb = bool(kwargs.get("eval_obb", True))

    if coco_evaluator is not None and eval_hbb:
        coco_evaluator.cleanup()

    metric_logger = MetricLogger(delimiter="  ")
    header = "Test:"
    iou_types = coco_evaluator.iou_types if coco_evaluator is not None else []

    gt: List[Dict[str, torch.Tensor]] = []
    preds: List[Dict[str, torch.Tensor]] = []
    obb_gt = []
    obb_preds = []

    output_dir = kwargs.get("output_dir", None)
    num_visualization_sample_batch = kwargs.get("num_visualization_sample_batch", 1)

    for i, (samples, targets) in enumerate(metric_logger.log_every(data_loader, 10, header)):
        if i < num_visualization_sample_batch and output_dir is not None and dist_utils.is_main_process():
            save_samples(samples, targets, output_dir, "val", normalized=False, box_fmt="xyxy")

        samples = samples.to(device)
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]
        outputs = model(samples)

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessor(outputs, orig_target_sizes)

        if eval_hbb and coco_evaluator is not None:
            res = {target["image_id"].item(): output for target, output in zip(targets, results)}
            coco_evaluator.update(res)

        for idx, (target, result) in enumerate(zip(targets, results)):
            labels = (
                torch.tensor([mscoco_category2label[int(x.item())] for x in result["labels"].flatten()])
                .to(result["labels"].device)
                .reshape(result["labels"].shape)
            ) if postprocessor.remap_mscoco_category else result["labels"]

            if eval_hbb:
                gt.append({
                    "boxes": scale_boxes(
                        target["boxes"],
                        (target["orig_size"][1], target["orig_size"][0]),
                        (samples[idx].shape[-1], samples[idx].shape[-2]),
                    ),
                    "labels": target["labels"],
                })
                preds.append({"boxes": result["boxes"], "labels": labels, "scores": result["scores"]})

            if eval_obb:
                obb_gt.append({
                    "rboxes": _scale_target_rboxes(target["boxes"], target["orig_size"].detach().cpu().numpy()),
                    "labels": target["labels"].detach().cpu().numpy().astype(np.int64),
                })
                pred_rboxes = result.get("rboxes", None)
                if pred_rboxes is not None:
                    obb_preds.append({
                        "rboxes": pred_rboxes.detach().cpu().numpy().astype(np.float64),
                        "labels": labels.detach().cpu().numpy().astype(np.int64),
                        "scores": result["scores"].detach().cpu().numpy().astype(np.float64),
                    })
                else:
                    obb_preds.append({
                        "rboxes": np.zeros((0, 5), dtype=np.float64),
                        "labels": np.zeros((0,), dtype=np.int64),
                        "scores": np.zeros((0,), dtype=np.float64),
                    })

    stats = {}

    if eval_hbb:
        metrics = Validator(gt, preds).compute_metrics()
        print("Metrics:", metrics)
        if use_wandb:
            metrics = {f"metrics/{k}": v for k, v in metrics.items()}
            metrics["epoch"] = epoch
            wandb.log(metrics)

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    if eval_hbb and coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()
        coco_evaluator.accumulate()
        coco_evaluator.summarize()

        if "bbox" in iou_types:
            bbox_eval = coco_evaluator.coco_eval["bbox"]
            stats["coco_eval_bbox"] = bbox_eval.stats.tolist()
            per_class_ap50, map50 = _compute_coco_ap50_per_class(bbox_eval, mscoco_category2name)
            _print_ap50_table(per_class_ap50, map50, title="DroneVehicle HBB AP@0.5 per class")
            stats["mAP50"] = map50
            for name, ap in per_class_ap50.items():
                stats[f"AP50_{name}"] = ap
        if "segm" in iou_types:
            stats["coco_eval_masks"] = coco_evaluator.coco_eval["segm"].stats.tolist()

    if eval_obb:
        obb_per_class, obb_map50 = _compute_obb_ap50(obb_gt, obb_preds, mscoco_category2name, iou_thr=0.5)
        _print_ap50_table(obb_per_class, obb_map50, title="DroneVehicle OBB AP@0.5 per class")
        stats["obb_mAP50"] = obb_map50
        for name, ap in obb_per_class.items():
            stats[f"obb_AP50_{name}"] = ap

    return stats, coco_evaluator
