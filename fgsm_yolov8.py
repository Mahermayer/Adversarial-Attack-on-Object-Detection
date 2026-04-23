#!/usr/bin/env python3
"""FGSM attack helper for YOLOv8 using Ultralytics training loss heads."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Dict, Tuple

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.utils.loss import v8DetectionLoss


class YoloV8FGSM:
    """One-step FGSM attack builder for YOLOv8 detection models."""

    def __init__(
        self,
        yolo_model,
        device: torch.device,
        epsilon: float = 4.0 / 255.0,
        box_gain: float = 7.5,
        cls_gain: float = 0.5,
        dfl_gain: float = 1.5,
        pseudo_conf: float = 0.25,
        pseudo_max_det: int = 30,
    ):
        self.yolo_model = yolo_model
        self.model = yolo_model.model
        self.device = device
        self.epsilon = float(epsilon)
        self.pseudo_conf = float(pseudo_conf)
        self.pseudo_max_det = int(max(1, pseudo_max_det))
        self.pseudo_yolo = self._build_pseudo_yolo()

        if isinstance(self.model.args, dict):
            args = dict(self.model.args)
            args.setdefault("box", box_gain)
            args.setdefault("cls", cls_gain)
            args.setdefault("dfl", dfl_gain)
            self.model.args = SimpleNamespace(**args)
        else:
            if not hasattr(self.model.args, "box"):
                self.model.args.box = box_gain
            if not hasattr(self.model.args, "cls"):
                self.model.args.cls = cls_gain
            if not hasattr(self.model.args, "dfl"):
                self.model.args.dfl = dfl_gain

        self.loss_fn = v8DetectionLoss(self.model)
        self.set_loss_gains(box_gain=box_gain, cls_gain=cls_gain, dfl_gain=dfl_gain)

    def _build_pseudo_yolo(self):
        ckpt_path = getattr(self.yolo_model, "ckpt_path", None) or self.yolo_model.overrides.get("model")
        if not ckpt_path:
            return self.yolo_model
        try:
            return YOLO(ckpt_path).to(self.device)
        except Exception:
            return self.yolo_model

    def set_loss_gains(self, box_gain: float, cls_gain: float, dfl_gain: float) -> None:
        """Expose YOLOv8 loss-head gains (box/cls/dfl) for FGSM objective."""
        self.loss_fn.hyp.box = float(box_gain)
        self.loss_fn.hyp.cls = float(cls_gain)
        self.loss_fn.hyp.dfl = float(dfl_gain)

    def _build_pseudo_batch(self, image_bgr: np.ndarray) -> Dict[str, torch.Tensor]:
        # Use model self-detections as pseudo labels. Convert via NumPy to avoid
        # inference-mode tensors leaking into autograd graph construction.
        with torch.no_grad():
            result = self.pseudo_yolo.predict(
                image_bgr,
                conf=self.pseudo_conf,
                imgsz=image_bgr.shape[0],
                max_det=self.pseudo_max_det,
                verbose=False,
            )[0]

        boxes = result.boxes
        if len(boxes):
            cls_np = boxes.cls.clone().detach().cpu().numpy().astype(np.float32, copy=False)
            bboxes_np = boxes.xywhn.clone().detach().cpu().numpy().astype(np.float32, copy=False)
            cls = torch.from_numpy(cls_np).to(self.device)
            bboxes = torch.from_numpy(np.clip(bboxes_np, 0.0, 1.0)).to(self.device)
            batch_idx = torch.zeros((cls.shape[0],), dtype=torch.float32, device=self.device)
        else:
            cls = torch.zeros((0,), dtype=torch.float32, device=self.device)
            bboxes = torch.zeros((0, 4), dtype=torch.float32, device=self.device)
            batch_idx = torch.zeros((0,), dtype=torch.float32, device=self.device)

        return {
            "batch_idx": batch_idx,
            "cls": cls,
            "bboxes": bboxes,
        }

    def _to_tensor(self, image_bgr: np.ndarray) -> torch.Tensor:
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        x = torch.from_numpy(rgb.transpose(2, 0, 1)).to(self.device)
        x = x.unsqueeze(0).to(dtype=torch.float32) / 255.0
        x.requires_grad_(True)
        return x

    @staticmethod
    def _to_bgr_uint8(x: torch.Tensor) -> np.ndarray:
        x = x.detach().clamp(0.0, 1.0)[0].permute(1, 2, 0).cpu().numpy()
        rgb = (x * 255.0).round().astype(np.uint8)
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    def generate(self, image_bgr: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
        """Generate FGSM-perturbed image and return loss diagnostics."""
        if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
            raise ValueError("Expected HxWx3 BGR image")

        pseudo_batch = self._build_pseudo_batch(image_bgr)
        x = self._to_tensor(image_bgr)

        self.model.zero_grad(set_to_none=True)
        if x.grad is not None:
            x.grad.zero_()

        preds = self.model(x)
        total_loss, loss_items = self.loss_fn(preds, pseudo_batch)
        total_loss = total_loss.sum()
        total_loss.backward()

        grad_sign = x.grad.sign()
        x_adv = (x + self.epsilon * grad_sign).clamp(0.0, 1.0)
        adv_bgr = self._to_bgr_uint8(x_adv)

        diagnostics = {
            "total": float(total_loss.detach().item()),
            "box": float(loss_items[0].item()),
            "cls": float(loss_items[1].item()),
            "dfl": float(loss_items[2].item()),
            "targets": int(pseudo_batch["cls"].numel()),
        }
        return adv_bgr, diagnostics
