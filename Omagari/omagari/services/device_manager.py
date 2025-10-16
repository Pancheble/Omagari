# omagari/services/device_manager.py
"""
DeviceManager centralizes:
- CPU/GPU selection
- torch/cv2/BLAS thread counts
- environment variables
- building the device string used by YOLO service
- moving models across devices

No UI calls here.
"""

from __future__ import annotations
import os
import logging
from typing import Optional, Tuple

from ..domain import AppState
from ..config import (
    CPU_THREADS_DEFAULT, GPU_COUNT_DEFAULT, LOG_LEVEL
)

logger = logging.getLogger(__name__)
logger.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))


class DeviceManager:
    """Stateless utility facade; methods operate on AppState + optional model."""

    BLAS_ENV_KEYS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")

    # --------- Detection ---------
    
    @staticmethod
    def detect_cpu_threads() -> Tuple[int, int]:
        """Return (logical_max, physical) with best-effort fallbacks."""
        try:
            import psutil  # optional
            logical = psutil.cpu_count(logical=True) or 1
            physical = psutil.cpu_count(logical=False) or logical
        except Exception:
            logical = os.cpu_count() or 1
            physical = logical
        return max(1, int(logical)), max(1, int(physical))

    @staticmethod
    def available_gpu_count() -> int:
        try:
            import torch
            return torch.cuda.device_count() if torch.cuda.is_available() else 0
        except Exception:
            return 0

    # --------- Threads / ENV ---------
    @classmethod
    def apply_threading(cls, threads: int) -> None:
        """Set threading hints for torch, cv2, and BLAS family."""
        t = max(1, int(threads))
        try:
            import torch
            torch.set_num_threads(t)
            if hasattr(torch, "set_num_interop_threads"):
                torch.set_num_interop_threads(t)
            logger.debug("Set torch threads: %s (interop as available)", t)
        except Exception as e:
            logger.debug("Torch thread setup skipped: %s", e)

        try:
            import cv2
            cv2.setNumThreads(t)
            logger.debug("Set OpenCV threads: %s", t)
        except Exception as e:
            logger.debug("OpenCV thread setup skipped: %s", e)

        for k in cls.BLAS_ENV_KEYS:
            os.environ[k] = str(t)

    # --------- Device string building ---------
    @classmethod
    def device_string(cls, state: AppState) -> str:
        """
        Return 'cpu' or a comma-joined CUDA index string like '0,1'.
        Clamps GPU count to availability.
        """
        mode = (state.yolo_device_mode or "CPU").upper()
        if mode == "CPU":
            return "cpu"
        avail = cls.available_gpu_count()
        if avail <= 0:
            return "cpu"
        want = max(1, int(state.yolo_gpu_count or GPU_COUNT_DEFAULT))
        count = min(want, avail)
        return ",".join(str(i) for i in range(count))

    @classmethod
    def pick_best_device(cls, state: AppState) -> str:
        """Simple helper honoring user intent: GPU if requested and available."""
        if (state.yolo_device_mode or "CPU").upper() == "GPU" and cls.available_gpu_count() > 0:
            return "0"
        return "cpu"

    # --------- Model moves ---------
    @classmethod
    def move_model_to_device(cls, model, device: str) -> None:
        """Best-effort move for Ultralytics model wrappers or raw torch nn.Module."""
        if model is None:
            return
        try:
            model.to(device)
            return
        except Exception:
            pass
        try:
            inner = getattr(model, "model", None)
            if inner is not None:
                inner.to(device)
        except Exception:
            logger.debug("Model move fallback failed", exc_info=True)

    @staticmethod
    def model_device_string(model) -> str:
        """Inspect underlying torch parameter device when possible."""
        try:
            m = getattr(model, "model", model)
            p = next(m.parameters(), None)
            return str(p.device) if p is not None else "unknown"
        except Exception:
            return "unknown"

    # --------- High-level apply ---------
    @classmethod
    def apply_device_now(cls, state: AppState, model=None) -> str:
        """
        Apply CPU threading if on CPU, move model to target device,
        and return a short status string for logging.
        """
        dev_str = cls.device_string(state)
        if dev_str == "cpu":
            cls.apply_threading(int(state.yolo_cpu_threads or CPU_THREADS_DEFAULT))
        cls.move_model_to_device(model, "cpu" if dev_str == "cpu" else dev_str.split(",")[0])
        mdev = cls.model_device_string(model) if model is not None else "n/a"
        # best-effort: report actual library thread settings
        try:
            import torch
            th = getattr(torch, "get_num_threads", lambda: -1)()
            ith = getattr(torch, "get_num_interop_threads", lambda: -1)()
        except Exception:
            th, ith = -1, -1
        try:
            import cv2
            cv_th = cv2.getNumThreads()
        except Exception:
            cv_th = -1
        msg = f"device={dev_str} model={mdev} torch_th={th}/{ith} cv2_th={cv_th}"
        logger.info("Device applied: %s", msg)
        return msg
