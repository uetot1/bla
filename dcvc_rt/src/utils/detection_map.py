"""Ground-truth object-detection mAP for VCM evaluation."""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch


IOU_THRESHOLDS = np.arange(0.50, 0.96, 0.05)
RECALL_THRESHOLDS = np.linspace(0.0, 1.0, 101)


def box_iou(boxes_a: torch.Tensor, boxes_b: torch.Tensor) -> torch.Tensor:
    if boxes_a.numel() == 0 or boxes_b.numel() == 0:
        return torch.empty((len(boxes_a), len(boxes_b)), dtype=torch.float32)
    top_left = torch.maximum(boxes_a[:, None, :2], boxes_b[None, :, :2])
    bottom_right = torch.minimum(boxes_a[:, None, 2:], boxes_b[None, :, 2:])
    intersection = (bottom_right - top_left).clamp_min(0).prod(dim=-1)
    area_a = (boxes_a[:, 2:] - boxes_a[:, :2]).clamp_min(0).prod(dim=-1)
    area_b = (boxes_b[:, 2:] - boxes_b[:, :2]).clamp_min(0).prod(dim=-1)
    union = area_a[:, None] + area_b[None, :] - intersection
    return intersection / union.clamp_min(1e-12)


class DetectionMAP:
    """Accumulate detections and compute 101-point interpolated AP."""

    def __init__(self):
        self.ground_truth: dict[int, dict[int, torch.Tensor]] = defaultdict(dict)
        self.predictions: dict[int, list[tuple[float, int, torch.Tensor]]] = defaultdict(list)
        self.image_ids: set[int] = set()

    def add(
        self,
        image_id: int,
        predicted_boxes: torch.Tensor,
        predicted_scores: torch.Tensor,
        predicted_classes: torch.Tensor,
        target_boxes: torch.Tensor,
        target_classes: torch.Tensor,
    ) -> None:
        if image_id in self.image_ids:
            raise ValueError(f"Duplicate image_id: {image_id}")
        self.image_ids.add(image_id)

        predicted_boxes = predicted_boxes.detach().cpu().float()
        predicted_scores = predicted_scores.detach().cpu().float()
        predicted_classes = predicted_classes.detach().cpu().long()
        target_boxes = target_boxes.detach().cpu().float()
        target_classes = target_classes.detach().cpu().long()

        for class_id in target_classes.unique().tolist():
            mask = target_classes == class_id
            self.ground_truth[int(class_id)][image_id] = target_boxes[mask]
        for box, score, class_id in zip(
            predicted_boxes,
            predicted_scores,
            predicted_classes,
        ):
            self.predictions[int(class_id)].append((float(score), image_id, box))

    def _average_precision(self, class_id: int, iou_threshold: float) -> float:
        targets = self.ground_truth[class_id]
        target_count = sum(len(boxes) for boxes in targets.values())
        if target_count == 0:
            raise ValueError(f"Class {class_id} has no ground-truth objects")

        predictions = sorted(
            self.predictions.get(class_id, []),
            key=lambda item: item[0],
            reverse=True,
        )
        matched = {
            image_id: torch.zeros(len(boxes), dtype=torch.bool)
            for image_id, boxes in targets.items()
        }
        true_positives = np.zeros(len(predictions), dtype=np.float64)
        false_positives = np.zeros(len(predictions), dtype=np.float64)

        for index, (_, image_id, predicted_box) in enumerate(predictions):
            image_targets = targets.get(image_id)
            if image_targets is None or len(image_targets) == 0:
                false_positives[index] = 1.0
                continue

            overlaps = box_iou(predicted_box.unsqueeze(0), image_targets)[0]
            best_iou, target_index = overlaps.max(dim=0)
            target_index = int(target_index)
            if best_iou >= iou_threshold and not matched[image_id][target_index]:
                true_positives[index] = 1.0
                matched[image_id][target_index] = True
            else:
                false_positives[index] = 1.0

        cumulative_tp = np.cumsum(true_positives)
        cumulative_fp = np.cumsum(false_positives)
        recall = cumulative_tp / target_count
        precision = cumulative_tp / np.maximum(cumulative_tp + cumulative_fp, 1e-12)
        interpolated = [
            precision[recall >= recall_threshold].max()
            if np.any(recall >= recall_threshold)
            else 0.0
            for recall_threshold in RECALL_THRESHOLDS
        ]
        return float(np.mean(interpolated))

    def compute(self) -> dict[str, float | int | dict[str, float]]:
        class_ids = sorted(self.ground_truth)
        if not class_ids:
            raise RuntimeError("No ground-truth objects were accumulated")

        per_threshold = {}
        for threshold in IOU_THRESHOLDS:
            class_aps = [
                self._average_precision(class_id, float(threshold))
                for class_id in class_ids
            ]
            per_threshold[f"{threshold:.2f}"] = float(np.mean(class_aps))

        return {
            "map50": per_threshold["0.50"],
            "map5095": float(np.mean(list(per_threshold.values()))),
            "evaluated_images": len(self.image_ids),
            "evaluated_classes": len(class_ids),
            "map_by_iou": per_threshold,
        }
