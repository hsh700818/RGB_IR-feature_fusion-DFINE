"""
D-FINE: Redefine Regression Task of DETRs as Fine-grained Distribution Refinement
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright (c) 2023 lyuwenyu. All Rights Reserved.
"""

import datetime
import json
import time

import torch

from ..misc import dist_utils, stats
from ._solver import BaseSolver
from .det_engine import evaluate, train_one_epoch


def _metric_first_value(value):
    """Return a scalar value for best-checkpoint comparison.

    COCO metrics are lists, for example coco_eval_bbox[0].  Custom metrics such
    as mAP50 and obb_mAP50 are floats.  The solver must support both forms.
    """
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return float(value.item())
        return float(value.flatten()[0].item())
    if isinstance(value, (list, tuple)):
        if len(value) == 0:
            return 0.0
        return _metric_first_value(value[0])
    try:
        return float(value)
    except Exception:
        return 0.0


def _iter_metric_values(value):
    """Yield scalar values for TensorBoard/W&B logging."""
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().flatten().tolist()
    if isinstance(value, (list, tuple)):
        for item in value:
            yield _metric_first_value(item)
    else:
        yield _metric_first_value(value)


class DetSolver(BaseSolver):
    def fit(self):
        self.train()
        args = self.cfg
        metric_names = ["AP50:95", "AP50", "AP75", "APsmall", "APmedium", "APlarge"]
        eval_freq = int(getattr(args, "eval_freq", 1))
        eval_freq = max(eval_freq, 1)
        primary_metric = getattr(args, "primary_metric", "obb_mAP50")

        if self.use_wandb:
            import wandb

            wandb.init(
                project=args.yaml_cfg["project_name"],
                name=args.yaml_cfg["exp_name"],
                config=args.yaml_cfg,
            )
            wandb.watch(self.model)

        n_parameters, model_stats = stats(self.cfg)
        print(model_stats)
        print("-" * 42 + "Start training" + "-" * 43)
        top1 = 0.0
        best_stat = {"epoch": -1}

        if self.last_epoch > 0:
            module = self.ema.module if self.ema else self.model
            test_stats, coco_evaluator = evaluate(
                module,
                self.criterion,
                self.postprocessor,
                self.val_dataloader,
                self.evaluator,
                self.device,
                self.last_epoch,
                self.use_wandb,
                output_dir=self.output_dir,
            )
            for k, v in test_stats.items():
                metric_value = _metric_first_value(v)
                best_stat["epoch"] = self.last_epoch
                best_stat[k] = metric_value
                if k == primary_metric or primary_metric not in test_stats:
                    top1 = metric_value
                print(f"best_stat: {best_stat}")

        best_stat_print = best_stat.copy()
        start_time = time.time()
        start_epoch = self.last_epoch + 1

        for epoch in range(start_epoch, args.epochs):
            self.train_dataloader.set_epoch(epoch)
            if dist_utils.is_dist_available_and_initialized():
                self.train_dataloader.sampler.set_epoch(epoch)

            if epoch == self.train_dataloader.collate_fn.stop_epoch:
                self.load_resume_state(str(self.output_dir / "best_stg1.pth"))
                if self.ema:
                    self.ema.decay = self.train_dataloader.collate_fn.ema_restart_decay
                    print(f"Refresh EMA at epoch {epoch} with decay {self.ema.decay}")

            train_stats = train_one_epoch(
                self.model,
                self.criterion,
                self.train_dataloader,
                self.optimizer,
                self.device,
                epoch,
                epochs=args.epochs,
                max_norm=args.clip_max_norm,
                print_freq=args.print_freq,
                ema=self.ema,
                scaler=self.scaler,
                lr_warmup_scheduler=self.lr_warmup_scheduler,
                writer=self.writer,
                use_wandb=self.use_wandb,
                output_dir=self.output_dir,
            )

            if self.lr_warmup_scheduler is None or self.lr_warmup_scheduler.finished():
                self.lr_scheduler.step()

            self.last_epoch += 1

            if self.output_dir and epoch < self.train_dataloader.collate_fn.stop_epoch:
                checkpoint_paths = [self.output_dir / "last.pth"]
                if (epoch + 1) % args.checkpoint_freq == 0:
                    checkpoint_paths.append(self.output_dir / f"checkpoint{epoch:04}.pth")
                for checkpoint_path in checkpoint_paths:
                    dist_utils.save_on_master(self.state_dict(), checkpoint_path)

            do_eval = ((epoch + 1) % eval_freq == 0) or ((epoch + 1) == args.epochs)
            test_stats = {}
            coco_evaluator = None

            if do_eval:
                print(f"Start evaluation at epoch {epoch}...")
                module = self.ema.module if self.ema else self.model
                test_stats, coco_evaluator = evaluate(
                    module,
                    self.criterion,
                    self.postprocessor,
                    self.val_dataloader,
                    self.evaluator,
                    self.device,
                    epoch,
                    self.use_wandb,
                    output_dir=self.output_dir,
                )

                compare_key = primary_metric if primary_metric in test_stats else None
                if compare_key is None:
                    compare_key = "coco_eval_bbox" if "coco_eval_bbox" in test_stats else next(iter(test_stats), None)

                current_primary = _metric_first_value(test_stats[compare_key]) if compare_key else 0.0
                is_best = current_primary > top1
                if is_best:
                    best_stat_print["epoch"] = epoch
                    top1 = current_primary
                    if self.output_dir:
                        if epoch >= self.train_dataloader.collate_fn.stop_epoch:
                            dist_utils.save_on_master(self.state_dict(), self.output_dir / "best_stg2.pth")
                        else:
                            dist_utils.save_on_master(self.state_dict(), self.output_dir / "best_stg1.pth")

                for k, v in test_stats.items():
                    metric_value = _metric_first_value(v)
                    if self.writer and dist_utils.is_main_process():
                        for i, item in enumerate(_iter_metric_values(v)):
                            self.writer.add_scalar(f"Test/{k}_{i}", item, epoch)

                    if k in best_stat:
                        if metric_value > best_stat[k]:
                            best_stat["epoch"] = epoch
                        best_stat[k] = max(best_stat[k], metric_value)
                    else:
                        best_stat.setdefault("epoch", epoch)
                        best_stat[k] = metric_value

                    best_stat_print[k] = best_stat[k]

                print(f"best_stat: {best_stat_print}")

                if epoch >= self.train_dataloader.collate_fn.stop_epoch and not is_best:
                    if self.ema:
                        self.ema.decay -= 0.0001
                        self.load_resume_state(str(self.output_dir / "best_stg1.pth"))
                        print(f"Refresh EMA at epoch {epoch} with decay {self.ema.decay}")
            else:
                next_eval = epoch + (eval_freq - ((epoch + 1) % eval_freq))
                print(f"Skip evaluation at epoch {epoch}. Next evaluation epoch: {next_eval}")

            log_stats = {
                **{f"train_{k}": v for k, v in train_stats.items()},
                **{f"test_{k}": v for k, v in test_stats.items()},
                "epoch": epoch,
                "n_parameters": n_parameters,
            }

            if self.use_wandb and do_eval:
                import wandb

                wandb_logs = {"epoch": epoch}
                if "coco_eval_bbox" in test_stats:
                    for idx, metric_name in enumerate(metric_names):
                        if idx < len(test_stats["coco_eval_bbox"]):
                            wandb_logs[f"metrics/{metric_name}"] = test_stats["coco_eval_bbox"][idx]
                for k, v in test_stats.items():
                    if not isinstance(v, (list, tuple)):
                        wandb_logs[f"metrics/{k}"] = _metric_first_value(v)
                wandb.log(wandb_logs)

            if self.output_dir and dist_utils.is_main_process():
                with (self.output_dir / "log.txt").open("a") as f:
                    f.write(json.dumps(log_stats) + "\n")

                if coco_evaluator is not None:
                    (self.output_dir / "eval").mkdir(exist_ok=True)
                    if "bbox" in coco_evaluator.coco_eval:
                        filenames = ["latest.pth"]
                        if epoch % 50 == 0:
                            filenames.append(f"{epoch:03}.pth")
                        for name in filenames:
                            torch.save(
                                coco_evaluator.coco_eval["bbox"].eval,
                                self.output_dir / "eval" / name,
                            )

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print("Training time {}".format(total_time_str))

    def val(self):
        self.eval()

        module = self.ema.module if self.ema else self.model
        test_stats, coco_evaluator = evaluate(
            module,
            self.criterion,
            self.postprocessor,
            self.val_dataloader,
            self.evaluator,
            self.device,
            epoch=-1,
            use_wandb=False,
            output_dir=self.output_dir,
        )

        if self.output_dir:
            dist_utils.save_on_master(
                coco_evaluator.coco_eval["bbox"].eval, self.output_dir / "eval.pth"
            )

        return
