from __future__ import annotations

import logging
import os
from collections.abc import Mapping
from typing import Any

import pandas as pd
import torch

import wandb
from connito.shared.config import MinerConfig, ValidatorConfig

from connito.shared.telemetry import (
    MINER_TRAINING_LOSS,
    MINER_LEARNING_RATE,
    MINER_LOCAL_STEP_RATE,
    MINER_GRAD_NORM,
    MINER_TOKENS_PER_SEC,
    MINER_GRAD_ACCUM_STEPS,
    MOE_AUX_LOSS,
    VALIDATOR_EVAL_LOSS,
    MINER_PERPLEXITY,
    MINER_TOTAL_TOKENS,
    MINER_TOTAL_SAMPLES,
    MINER_STEP_TIME_HOURS,
    MINER_TOTAL_TRAINING_TIME_HOURS,
    MINER_PARAM_SUM
)

logger = logging.getLogger(__name__)


class MetricLogger:
    """
    Write metrics locally (CSV) and optionally to Weights & Biases.

    Parameters
    ----------
    config : MinerConfig
        Must provide:
          - metric_path: str (CSV file path)
          - log_wandb: bool
          - wandb_project_name: str
          - run_name: str
          - model_path: str
    """

    def __init__(self, config: MinerConfig | ValidatorConfig, rank: int = 0, validation: bool = False) -> None:
        self.csv_path: str = config.log.metric_path
        self.log_wandb: bool = bool(config.log.log_wandb)
        self.validation = validation

        # Ensure the metrics directory exists.
        metrics_dir = os.path.dirname(self.csv_path) or "."
        os.makedirs(metrics_dir, exist_ok=True)

        run_name = f"[full] {config.run.run_name}" if validation else f"[partial] rank{rank}-{config.run.run_name}"
        self.wandb_run: wandb.sdk.wandb_run.Run | None = None

        if config.log.wandb_resume:
            if validation:
                run_id = config.log.wandb_full_id
            else:
                run_id = config.log.wandb_partial_id[rank]
        else:
            run_id = None

        if self.log_wandb:
            try:
                self.wandb_run = wandb.init(
                    entity="isabella_cl-cruciblelabs",
                    project=f"subnet-expert-{config.log.wandb_project_name}",
                    name=run_name,
                    tags=[config.run.run_name],
                    id=run_id,
                    resume="allow",
                    config=config.__dict__,
                )
            except Exception as e:
                logger.warning(f"W&B init failed, disabling W&B logging: {e}")
                self.log_wandb = False
                self.wandb_run = None

    # -------- public API --------
    def close(self) -> None:
        """Finish the W&B run (if enabled)."""
        if self.wandb_run is not None:
            try:
                self.wandb_run.finish()
            except Exception as e:
                logger.warning(f"W&B finish failed: {e}")
            finally:
                self.wandb_run = None

    # -------- helpers --------
    def _wandb_log(self, metrics: dict[str, Any]) -> None:
        try:
            self.wandb_run.log(metrics)

            if "val_loss" in metrics:
                self.wandb_run.alert(title=f"Success log: validation {self.validation}", text=f"{metrics}")

        except Exception as e:
            logger.warning(f"W&B log failed (will continue locally): {e}")

    def log(self, metrics: Mapping[str, Any], print_log=True) -> None:
        """
        Log a single metrics dict to CSV (always) and W&B (optional).

        Any torch tensors will be converted to Python scalars (0-D) or lists (N-D).
        Sequences (list/tuple) are stored as the first element if len==1, else as stringified lists.
        """

        flat = self._flatten_metrics(metrics)

        if print_log:
            logger.info("Metric log: %s", {k: round(v, 4) if isinstance(v, float) else v for k, v in flat.items()})

        # Prometheus telemetry updates
        try:
            if "loss" in flat and isinstance(flat["loss"], (int, float)):
                group_name = str(flat.get("expert_group_name", "unknown_group"))
                MINER_TRAINING_LOSS.labels(expert_group=group_name).set(flat["loss"])
            if "eval_loss" in flat and isinstance(flat["eval_loss"], (int, float)):
                group_name = str(flat.get("expert_group_name", "unknown_group"))
                VALIDATOR_EVAL_LOSS.labels(expert_group=group_name).set(flat["eval_loss"])
            if "lr" in flat and isinstance(flat["lr"], (int, float)):
                MINER_LEARNING_RATE.set(flat["lr"])
            if "step_rate" in flat and isinstance(flat["step_rate"], (int, float)):
                MINER_LOCAL_STEP_RATE.set(flat["step_rate"])
            if "tokens_per_second" in flat and isinstance(flat["tokens_per_second"], (int, float)):
                MINER_TOKENS_PER_SEC.set(flat["tokens_per_second"])
            if "grad_norm" in flat and isinstance(flat["grad_norm"], (int, float)):
                MINER_GRAD_NORM.set(flat["grad_norm"])
            if "gradient_accumulation_steps" in flat and isinstance(flat["gradient_accumulation_steps"], (int, float)):
                MINER_GRAD_ACCUM_STEPS.set(flat["gradient_accumulation_steps"])
            if "aux_loss" in flat and isinstance(flat["aux_loss"], (int, float)):
                MOE_AUX_LOSS.set(flat["aux_loss"])
            if "perplexity" in flat and isinstance(flat["perplexity"], (int, float)):
                MINER_PERPLEXITY.set(flat["perplexity"])
            if "total_tokens" in flat and isinstance(flat["total_tokens"], (int, float)):
                MINER_TOTAL_TOKENS.set(flat["total_tokens"])
            if "total_samples" in flat and isinstance(flat["total_samples"], (int, float)):
                MINER_TOTAL_SAMPLES.set(flat["total_samples"])
            if "inner_step_time_hours" in flat and isinstance(flat["inner_step_time_hours"], (int, float)):
                MINER_STEP_TIME_HOURS.set(flat["inner_step_time_hours"])
            if "total_training_time_hours" in flat and isinstance(flat["total_training_time_hours"], (int, float)):
                MINER_TOTAL_TRAINING_TIME_HOURS.set(flat["total_training_time_hours"])
            if "param_sum" in flat and isinstance(flat["param_sum"], (int, float)):
                MINER_PARAM_SUM.set(flat["param_sum"])
        except Exception as e:
            logger.debug(f"Failed to push stats to telemetry: {e}")

        self._local_log(flat)
        if self.log_wandb and self.wandb_run is not None:
            self._wandb_log(flat)

    def _local_log(self, metrics: dict):
        # Flatten list/tensor values
        flat_metrics = {}
        for k, v in metrics.items():
            if isinstance(v, list | tuple):
                flat_metrics[k] = v[0] if len(v) == 1 else str(v)
            elif torch.is_tensor(v):
                flat_metrics[k] = v.item() if v.ndim == 0 else v.detach().cpu().tolist()
            else:
                flat_metrics[k] = v

        new_row = pd.DataFrame([flat_metrics])

        if not os.path.exists(self.csv_path):
            # First write: save with header
            new_row.to_csv(self.csv_path, index=False)
        else:
            # Load existing CSV and merge schemas.
            try:
                existing = pd.read_csv(self.csv_path)
            except Exception as e:
                logger.warning(f"Failed to read existing metrics CSV, rewriting header: {e}")
                new_row.to_csv(self.csv_path, index=False)
                self._fsync_if_supported()
                return

            # Add any new columns to existing (as empty) and to new_row (for missing in new).
            for col in new_row.columns:
                if col not in existing.columns:
                    existing[col] = ""  # backfill empty values for prior rows
            for col in existing.columns:
                if col not in new_row.columns:
                    new_row[col] = ""  # ensure consistent column order

            # Reorder new_row to match existing column order and append.

            new_row = new_row.reindex(columns=existing.columns)
            updated = pd.concat([existing, new_row], ignore_index=True)
            updated.to_csv(self.csv_path, index=False)

        self._fsync_if_supported()

    @staticmethod
    def _flatten_metrics(metrics: Mapping[str, Any]) -> dict[str, Any]:
        """
        Convert metrics dict into CSV/W&B-safe values.

        Rules
        -----
        * torch.Tensor:
            - 0-D -> .item()
            - N-D -> .detach().cpu().tolist()
        * list/tuple:
            - len == 1 -> first element
            - else -> stringified list (to keep CSV rectangular)
        * everything else -> as-is
        """
        flat: dict[str, Any] = {}
        for k, v in metrics.items():
            if torch.is_tensor(v):
                flat[k] = v.item() if v.ndim == 0 else v.detach().cpu().tolist()
            elif isinstance(v, list | tuple):
                flat[k] = v[0] if len(v) == 1 else str(v)
            else:
                flat[k] = v
        return flat

    @staticmethod
    def _fsync_if_supported() -> None:
        """
        Force a disk flush if the OS provides `os.sync` (Linux).
        Helpful for real-time monitoring tailing the CSV.
        """
        if hasattr(os, "sync"):
            try:
                os.sync()
            except Exception:
                # Best-effort; ignore if the platform denies it.
                pass
