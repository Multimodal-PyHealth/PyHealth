"""Self-supervised pretraining trainer.

A lightweight trainer specialized for reconstruction / latent-prediction
objectives.  It is intentionally separate from :class:`pyhealth.trainer.Trainer`
because SSL has no labels, validation metrics, or classification head.

Supports single-GPU and multi-GPU (DDP) training.  For DDP, launch with::

    torchrun --nproc_per_node=4 scripts/pretrain_ssl.py --config ...
"""

from __future__ import annotations

import json
import logging
import math
import os
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Type

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Optimizer
from torch.utils.data import DataLoader, DistributedSampler, IterableDataset
from tqdm import tqdm
from tqdm.autonotebook import trange

from pyhealth import _wandb
from pyhealth.utils import create_directory

logger = logging.getLogger(__name__)


def set_logger(log_path: str) -> None:
    create_directory(log_path)
    log_filename = os.path.join(log_path, "log.txt")
    handler = logging.FileHandler(log_filename)
    formatter = logging.Formatter("%(asctime)s %(message)s", "%Y-%m-%d %H:%M:%S")
    handler.setFormatter(formatter)
    logger.addHandler(handler)


def _vram_stats(device: str) -> Dict[str, float]:
    if not torch.cuda.is_available() or not str(device).startswith("cuda"):
        return {}
    allocated = torch.cuda.memory_allocated(device) / 1024**2
    peak = torch.cuda.max_memory_allocated(device) / 1024**2
    return {"vram_allocated_mb": allocated, "vram_peak_mb": peak}


def _is_ddp() -> bool:
    return (
        "RANK" in os.environ
        and "WORLD_SIZE" in os.environ
        and int(os.environ["WORLD_SIZE"]) > 1
    )


def _get_rank() -> int:
    return int(os.environ.get("RANK", 0))


def _get_local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", 0))


def _is_main_process() -> bool:
    return _get_rank() == 0


class PretrainTrainer:
    """Trainer for self-supervised pretraining objectives.

    Args:
        model: Pretraining model (MAE, SimMIM, I-JEPA, ...).
        device: Device to use.  Auto-detected if None.  Under DDP this is set
            automatically to the local GPU.
        enable_logging: Whether to write ``log.txt`` and ``metrics_history.json``.
            Only the main process writes logs.
        output_path: Root directory for checkpoints and logs.
        exp_name: Experiment subdirectory name.
        ema_update_fn: Optional callable invoked once per training step to
            update an EMA target network (used by I-JEPA).
        ema_update_every: Call ``ema_update_fn`` every N steps.  Default 1.
        use_ddp: If True, use DistributedDataParallel.  Defaults to True when
            launched via ``torchrun`` / ``RANK`` env vars.
    """

    def __init__(
        self,
        model: nn.Module,
        device: Optional[str] = None,
        enable_logging: bool = True,
        output_path: Optional[str] = None,
        exp_name: Optional[str] = None,
        ema_update_fn: Optional[Callable[[], None]] = None,
        ema_update_every: int = 1,
        use_ddp: Optional[bool] = None,
    ):
        self._ddp = use_ddp if use_ddp is not None else _is_ddp()
        if self._ddp:
            if not dist.is_initialized():
                dist.init_process_group("nccl" if torch.cuda.is_available() else "gloo")
            self.rank = _get_rank()
            self.world_size = int(os.environ["WORLD_SIZE"])
            device = f"cuda:{_get_local_rank()}" if torch.cuda.is_available() else "cpu"
        else:
            self.rank = 0
            self.world_size = 1
            if device is None:
                device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = model
        self.device = device
        self.ema_update_fn = ema_update_fn
        self.ema_update_every = ema_update_every

        if enable_logging and _is_main_process():
            if output_path is None:
                output_path = os.path.join(os.getcwd(), "output")
            if exp_name is None:
                exp_name = time.strftime("%Y%m%d-%H%M%S")
            self.exp_path = os.path.join(output_path, exp_name)
            set_logger(self.exp_path)
        else:
            self.exp_path = None

        self.model.to(self.device)
        if self._ddp:
            self.model = DDP(
                self.model,
                device_ids=[_get_local_rank()] if torch.cuda.is_available() else None,
                output_device=_get_local_rank() if torch.cuda.is_available() else None,
                find_unused_parameters=False,
            )

        if _is_main_process():
            logger.info(self.model)
            logger.info(f"Device: {self.device}")
            if self._ddp:
                logger.info(f"DDP world size: {self.world_size}")

    def train(
        self,
        train_dataloader: DataLoader,
        epochs: int = 10,
        optimizer_class: Type[Optimizer] = torch.optim.AdamW,
        optimizer_params: Optional[Dict[str, object]] = None,
        steps_per_epoch: Optional[int] = None,
        weight_decay: float = 0.05,
        max_grad_norm: Optional[float] = 1.0,
        scheduler: Optional[str] = None,
        warmup_steps: int = 0,
        save_every_n_epochs: int = 1,
        grad_accumulation_steps: int = 1,
        use_amp: bool = False,
        val_dataloader: Optional[DataLoader] = None,
        epoch_callback: Optional[Callable[[int, Dict], None]] = None,
    ) -> List[Dict[str, object]]:
        """Run SSL pretraining.

        Args:
            train_dataloader: Dataloader yielding batches compatible with the
                pretraining model's ``forward``.  Under DDP this should be
                paired with a ``DistributedSampler`` by the caller; if a plain
                sampler is detected, the trainer wraps it automatically.
            epochs: Number of epochs.
            optimizer_class: Optimizer class.
            optimizer_params: Optimizer kwargs.  Defaults to ``{"lr": 1e-4}``.
            steps_per_epoch: If None, uses ``len(train_dataloader)``.
            weight_decay: Weight decay.  Applied via param-group split.
            max_grad_norm: Gradient clipping.  None disables.
            scheduler: ``"cosine"`` or None.
            warmup_steps: Linear warmup steps.
            save_every_n_epochs: Save a checkpoint every N epochs.
            grad_accumulation_steps: Number of forward/backward steps to
                accumulate before an optimizer step.  Effective batch size is
                ``batch_size * world_size * grad_accumulation_steps``.
            use_amp: Use automatic mixed precision (torch.cuda.amp) on CUDA.

        Returns:
            List of per-epoch metric dicts (only from the main process when DDP).
        """
        if optimizer_params is None:
            optimizer_params = {"lr": 1e-4}

        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]

        def _decayed(n):
            return not any(nd in n for nd in no_decay)

        # Unwrap DDP module for parameter grouping if needed.
        raw_model = self.model.module if isinstance(self.model, DDP) else self.model
        param_groups = [
            {"params": [p for n, p in raw_model.named_parameters() if _decayed(n)],
             "weight_decay": weight_decay},
            {"params": [p for n, p in raw_model.named_parameters() if not _decayed(n)],
             "weight_decay": 0.0},
        ]
        optimizer = optimizer_class(param_groups, **optimizer_params)

        # Auto-wrap sampler if not already distributed. IterableDataset (e.g.
        # litdata StreamingDataset) shards across DDP ranks internally and cannot
        # take a sampler, so skip the wrap there.
        sampler = train_dataloader.sampler
        _iterable = isinstance(train_dataloader.dataset, IterableDataset)
        if self._ddp and not _iterable and not isinstance(sampler, DistributedSampler):
            train_dataloader = DataLoader(
                train_dataloader.dataset,
                batch_size=train_dataloader.batch_size,
                sampler=DistributedSampler(
                    train_dataloader.dataset,
                    num_replicas=self.world_size,
                    rank=self.rank,
                    shuffle=True,
                ),
                num_workers=train_dataloader.num_workers,
                collate_fn=train_dataloader.collate_fn,
                pin_memory=train_dataloader.pin_memory,
                drop_last=train_dataloader.drop_last,
            )

        total_steps = epochs * (steps_per_epoch or len(train_dataloader))
        # The scheduler steps once per *optimizer* step, not per micro-step, so
        # the horizon must be divided by the grad-accumulation factor.
        total_optim_steps = max(1, total_steps // max(1, grad_accumulation_steps))

        if scheduler == "cosine" or warmup_steps > 0:
            warmup = max(0, int(warmup_steps))

            def _lr_lambda(opt_step: int) -> float:
                # Linear warmup, then (optionally) cosine decay to ~0.
                if warmup > 0 and opt_step < warmup:
                    return (opt_step + 1) / warmup
                if scheduler == "cosine":
                    progress = (opt_step - warmup) / max(1, total_optim_steps - warmup)
                    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
                return 1.0

            sched = torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda)
        else:
            sched = None

        use_amp = use_amp and torch.cuda.is_available()
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp) if use_amp else None

        if _is_main_process():
            logger.info("SSL pretraining:")
            logger.info(f"Batch size: {train_dataloader.batch_size}")
            logger.info(f"Optimizer: {optimizer_class}")
            logger.info(f"Optimizer params: {optimizer_params}")
            logger.info(f"Weight decay: {weight_decay}")
            logger.info(f"Max grad norm: {max_grad_norm}")
            logger.info(f"Epochs: {epochs}")
            logger.info(f"Grad accumulation steps: {grad_accumulation_steps}")
            logger.info(f"AMP: {use_amp}")
            logger.info(f"EMA update every: {self.ema_update_every} steps")

        data_iterator = iter(train_dataloader)
        if steps_per_epoch is None:
            steps_per_epoch = len(train_dataloader)
        global_step = 0       # counts micro (forward/backward) steps
        optimizer_step = 0    # counts actual optimizer updates
        metrics_history: List[Dict[str, object]] = []
        train_start = time.perf_counter()

        # Resume from a prior (e.g. preempted/requeued) run: restore model +
        # optimizer/scheduler state + epoch so it continues instead of restarting.
        start_epoch = 0
        if self.exp_path is not None:
            _ck = os.path.join(self.exp_path, "last.ckpt")
            _rs = os.path.join(self.exp_path, "_resume.pt")
            if os.path.isfile(_ck) and os.path.isfile(_rs):
                self.load_ckpt(_ck)
                _state = torch.load(_rs, map_location=self.device, weights_only=False)
                optimizer.load_state_dict(_state["optimizer"])
                if sched is not None and _state.get("sched") is not None:
                    sched.load_state_dict(_state["sched"])
                start_epoch = int(_state["epoch"]) + 1
                global_step = int(_state.get("global_step", 0))
                optimizer_step = int(_state.get("optimizer_step", 0))
                _mh = os.path.join(self.exp_path, "metrics_history.json")
                if os.path.isfile(_mh):
                    with open(_mh) as f:
                        metrics_history = json.load(f)[:start_epoch]
                if _is_main_process():
                    logger.info(f"Resuming from epoch {start_epoch}/{epochs}")
            elif os.path.isfile(_ck):
                # Partial run with a weights checkpoint but no optimizer state
                # (e.g. produced before atomic _resume.pt existed): load the
                # trained weights and infer the epoch from metrics_history so we
                # finish the remaining epochs instead of cold-restarting from 0.
                self.load_ckpt(_ck)
                _mh = os.path.join(self.exp_path, "metrics_history.json")
                if os.path.isfile(_mh):
                    with open(_mh) as f:
                        metrics_history = json.load(f)
                    start_epoch = len(metrics_history)
                if _is_main_process():
                    logger.info(f"Warm-resuming from last.ckpt at epoch "
                                f"{start_epoch}/{epochs} (no optimizer state)")

        # Opt-in W&B tracking (full runs only: exp_path is None under Optuna's
        # enable_logging=False, so per-trial training never spawns runs).
        wrun = None
        if self.exp_path is not None and _is_main_process():
            # Legible naming/config: exp dir basename is "{arch}_{method}_{task}_seed{N}".
            _exp_name = getattr(self, "exp_name", None) or os.path.basename(self.exp_path.rstrip("/"))
            _arch = _method = _task = _seed = None
            if _exp_name and "_seed" in _exp_name:
                _body, _, _seed = _exp_name.rpartition("_seed")
                _parts = _body.split("_", 2)
                if len(_parts) == 3:
                    _arch, _method, _task = _parts
            _mdl = self.model.module if hasattr(self.model, "module") else self.model  # unwrap DDP
            wrun = _wandb.init_run(
                config={"exp_name": _exp_name, "arch": _arch, "method": _method,
                        "task": _task, "seed": _seed, "epochs": epochs,
                        "lr": optimizer_params.get("lr"), "weight_decay": weight_decay,
                        "batch_size": getattr(train_dataloader, "batch_size", None),
                        "model": type(_mdl).__name__, "kind": "pretrain"},
                name=_exp_name, group=_task, job_type="pretrain",
                tags=["kind:pretrain", "stage:pretrain",
                      f"bb:{_arch}" if _arch else None,
                      f"mod:{_task}" if _task else None,
                      f"method:{_method}" if _method else None])

        epoch_iterator = tqdm(
            range(start_epoch, epochs),
            initial=start_epoch, total=epochs,
            desc="Pretrain epochs",
            unit="epoch",
            disable=not _is_main_process(),
        )
        for epoch in epoch_iterator:
            epoch_iterator.set_postfix_str(f"{epoch + 1}/{epochs}", refresh=False)
            if isinstance(train_dataloader.sampler, DistributedSampler):
                train_dataloader.sampler.set_epoch(epoch)

            self.model.train()
            if torch.cuda.is_available() and str(self.device).startswith("cuda"):
                torch.cuda.reset_peak_memory_stats(self.device)
            epoch_start = time.perf_counter()
            epoch_losses = []
            epoch_loss_dicts: List[Dict[str, float]] = []

            for _ in trange(
                steps_per_epoch,
                desc=f"Epoch {epoch + 1}/{epochs}",
                smoothing=0.05,
                leave=False,
                disable=not _is_main_process(),
            ):
                try:
                    data = next(data_iterator)
                except StopIteration:
                    data_iterator = iter(train_dataloader)
                    data = next(data_iterator)

                data = self._to_device(data)

                with torch.amp.autocast(device_type="cuda", enabled=use_amp):
                    output = self.model(**data)
                    loss = output["loss"] / grad_accumulation_steps

                if use_amp:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                if (global_step + 1) % grad_accumulation_steps == 0:
                    if max_grad_norm is not None:
                        if use_amp:
                            scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), max_grad_norm
                        )

                    if use_amp:
                        # Detect whether GradScaler actually applied the step
                        # (it skips on inf/NaN grads, shrinking the scale).
                        scale_before = scaler.get_scale()
                        scaler.step(optimizer)
                        scaler.update()
                        stepped = scaler.get_scale() >= scale_before
                    else:
                        optimizer.step()
                        stepped = True
                    optimizer.zero_grad()

                    # Only advance the schedule / EMA when a real update happened.
                    if stepped:
                        optimizer_step += 1
                        # Warmup + cosine decay are handled by the LambdaLR
                        # scheduler, which steps once per optimizer step.
                        if sched is not None:
                            sched.step()

                        if (
                            self.ema_update_fn is not None
                            and optimizer_step % self.ema_update_every == 0
                        ):
                            # Advance the EMA momentum schedule (cosine ->end)
                            # on the optimizer-step clock, then EMA-copy.
                            if hasattr(raw_model, "set_ema_decay"):
                                raw_model.set_ema_decay(optimizer_step, total_optim_steps)
                            # EMA function lives on the unwrapped model.
                            self.ema_update_fn()

                epoch_losses.append(loss.item() * grad_accumulation_steps)
                if "loss_dict" in output:
                    epoch_loss_dicts.append(output["loss_dict"])

                global_step += 1

            epoch_time = time.perf_counter() - epoch_start
            vram = _vram_stats(self.device)

            avg_loss_dict: Dict[str, float] = {}
            if epoch_loss_dicts:
                keys = set(k for d in epoch_loss_dicts for k in d)
                for k in keys:
                    vals = [d[k] for d in epoch_loss_dicts if k in d]
                    if vals:
                        avg_loss_dict[k] = sum(vals) / len(vals)

            epoch_record: Dict[str, object] = {
                "epoch": epoch,
                "global_step": global_step,
                "train_loss": sum(epoch_losses) / len(epoch_losses),
                "epoch_time_s": round(epoch_time, 3),
                "learning_rate": optimizer.param_groups[0]["lr"],
                **avg_loss_dict,
                **{f"train_{k}": v for k, v in vram.items()},
            }

            # Optional held-out SSL loss (used e.g. as the Optuna objective).
            if val_dataloader is not None:
                epoch_record.update(self._validate(val_dataloader, use_amp))

            if _is_main_process():
                logger.info(f"--- Pretrain epoch-{epoch}, step-{global_step} ---")
                logger.info(f"loss: {epoch_record['train_loss']:.4f}")
                if "val_loss" in epoch_record:
                    logger.info(f"val_loss: {epoch_record['val_loss']:.4f}")
                logger.info(f"epoch_time: {epoch_time:.2f}s")
                if vram:
                    logger.info(
                        f"vram_peak: {vram['vram_peak_mb']:.1f} MB  "
                        f"vram_current: {vram['vram_allocated_mb']:.1f} MB"
                    )

            metrics_history.append(epoch_record)
            _wandb.log(wrun, epoch_record, step=epoch)

            # Per-epoch hook (e.g. Optuna pruning); may raise to abort early.
            if epoch_callback is not None:
                epoch_callback(epoch, epoch_record)

            if _is_main_process() and self.exp_path is not None:
                self.save_ckpt(os.path.join(self.exp_path, "last.ckpt"))
                # resume state (optimizer/scheduler/epoch) — written atomically so
                # a preemption mid-write can't corrupt it.
                _rs = os.path.join(self.exp_path, "_resume.pt")
                torch.save({"epoch": epoch, "global_step": global_step,
                            "optimizer_step": optimizer_step,
                            "optimizer": optimizer.state_dict(),
                            "sched": sched.state_dict() if sched is not None else None},
                           _rs + ".tmp")
                os.replace(_rs + ".tmp", _rs)
                if (epoch + 1) % save_every_n_epochs == 0:
                    self.save_ckpt(
                        os.path.join(self.exp_path, f"epoch_{epoch + 1}.ckpt")
                    )
                history_path = os.path.join(self.exp_path, "metrics_history.json")
                with open(history_path, "w") as f:
                    json.dump(metrics_history, f, indent=2)

        if self._ddp:
            dist.barrier()

        if _is_main_process():
            total_time = time.perf_counter() - train_start
            logger.info(f"--- Pretraining complete: {total_time:.2f}s total ---")
        _wandb.finish(wrun)

        return metrics_history

    @torch.no_grad()
    def _validate(self, val_dataloader: DataLoader, use_amp: bool = False) -> Dict[str, float]:
        """Mean SSL loss (and per-component breakdown) over a held-out loader."""
        self.model.eval()
        losses: List[float] = []
        loss_dicts: List[Dict[str, float]] = []
        for data in val_dataloader:
            data = self._to_device(data)
            with torch.amp.autocast(device_type="cuda", enabled=use_amp):
                output = self.model(**data)
            losses.append(output["loss"].item())
            if "loss_dict" in output:
                loss_dicts.append(output["loss_dict"])
        self.model.train()
        if not losses:
            return {}
        record: Dict[str, float] = {"val_loss": sum(losses) / len(losses)}
        if loss_dicts:
            keys = set(k for d in loss_dicts for k in d)
            for k in keys:
                vals = [d[k] for d in loss_dicts if k in d]
                if vals:
                    record[f"val_{k}"] = sum(vals) / len(vals)
        return record

    def _to_device(self, data):
        """Recursively move tensors in a nested dict to the trainer device."""

        def _move(obj):
            if isinstance(obj, torch.Tensor):
                return obj.to(self.device)
            if isinstance(obj, dict):
                return {k: _move(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return type(obj)(_move(x) for x in obj)
            return obj

        return _move(data)

    def save_ckpt(self, ckpt_path: str) -> None:
        """Save model state dict and training metadata."""
        Path(ckpt_path).parent.mkdir(parents=True, exist_ok=True)
        raw_model = self.model.module if isinstance(self.model, DDP) else self.model
        torch.save(raw_model.state_dict(), ckpt_path)

    def load_ckpt(self, ckpt_path: str) -> None:
        """Load model state dict."""
        state_dict = torch.load(
            ckpt_path, map_location=self.device, weights_only=True
        )
        raw_model = self.model.module if isinstance(self.model, DDP) else self.model
        raw_model.load_state_dict(state_dict)
