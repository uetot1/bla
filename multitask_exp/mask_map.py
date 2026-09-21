"""Instance-mask mAP, computed exactly like the project's box mAP but on mask IoU.

Mirrors ``dcvc_rt/src/utils/detection_map.py`` step for step -- same IoU sweep,
same 101-point interpolated AP, same "classes with no ground truth are skipped"
rule -- so a mask number and a box number from this project mean the same thing
and can sit in the same table.

The one implementation difference is memory. Storing every mask until compute()
would cost gigabytes over a 600-frame sequence, so add() reduces each image to
its per-class IoU matrix immediately and keeps only floats. That is lossless for
AP: a prediction can only ever match ground truth inside its own image, so the
IoU matrices carry everything the greedy matcher reads.
"""

from collections import defaultdict

import numpy as np
import torch

from dcvc_rt.src.utils.detection_map import IOU_THRESHOLDS, RECALL_THRESHOLDS


def mask_iou(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """IoU between every predicted mask and every target mask. Both [n, H, W] boolean."""
    if predicted.numel() == 0 or target.numel() == 0:
        return torch.zeros((len(predicted), len(target)), dtype=torch.float32)
    a = predicted.flatten(1).float()
    b = target.flatten(1).float()
    intersection = a @ b.t()
    union = a.sum(1)[:, None] + b.sum(1)[None, :] - intersection
    return (intersection / union.clamp_min(1e-12)).float()


class MaskMAP:
    """Accumulate instance masks and compute 101-point interpolated AP on mask IoU."""

    def __init__(self):
        self.target_counts: dict[int, dict[int, int]] = defaultdict(dict)
        self.predictions: dict[int, list[tuple[float, int, int]]] = defaultdict(list)
        self.iou: dict[tuple[int, int], torch.Tensor] = {}
        self.image_ids: set[int] = set()

    def add(
        self,
        image_id: int,
        predicted_masks: torch.Tensor,
        predicted_scores: torch.Tensor,
        predicted_classes: torch.Tensor,
        target_masks: torch.Tensor,
        target_classes: torch.Tensor,
    ) -> None:
        if image_id in self.image_ids:
            raise ValueError(f"Duplicate image_id: {image_id}")
        self.image_ids.add(image_id)

        predicted_scores = predicted_scores.detach().cpu().float()
        predicted_classes = predicted_classes.detach().cpu().long()
        target_classes = target_classes.detach().cpu().long()
        if len(predicted_masks) != len(predicted_classes) or len(target_masks) != len(target_classes):
            raise ValueError("Mask count must match class count")

        classes = set(predicted_classes.tolist()) | set(target_classes.tolist())
        for class_id in classes:
            target_rows = target_classes == class_id
            predicted_rows = predicted_classes == class_id
            target_count = int(target_rows.sum())
            if target_count:
                self.target_counts[class_id][image_id] = target_count
            if not bool(predicted_rows.any()):
                continue
            self.iou[(class_id, image_id)] = mask_iou(
                predicted_masks[predicted_rows.to(predicted_masks.device)],
                target_masks[target_rows.to(target_masks.device)],
            ).cpu()
            for local_index, score in enumerate(predicted_scores[predicted_rows].tolist()):
                self.predictions[class_id].append((float(score), image_id, local_index))

    def _average_precision(self, class_id: int, iou_threshold: float) -> float:
        targets = self.target_counts[class_id]
        target_count = sum(targets.values())
        if target_count == 0:
            raise ValueError(f"Class {class_id} has no reference objects")

        predictions = sorted(self.predictions.get(class_id, []), key=lambda item: item[0],
                             reverse=True)
        matched = {image_id: torch.zeros(count, dtype=torch.bool)
                   for image_id, count in targets.items()}
        true_positives = np.zeros(len(predictions), dtype=np.float64)
        false_positives = np.zeros(len(predictions), dtype=np.float64)

        for index, (_, image_id, local_index) in enumerate(predictions):
            overlaps = self.iou.get((class_id, image_id))
            if image_id not in targets or overlaps is None or overlaps.shape[1] == 0:
                false_positives[index] = 1.0
                continue
            best_iou, target_index = overlaps[local_index].max(dim=0)
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
            if np.any(recall >= recall_threshold) else 0.0
            for recall_threshold in RECALL_THRESHOLDS
        ]
        return float(np.mean(interpolated))

    def compute(self) -> dict:
        class_ids = sorted(self.target_counts)
        if not class_ids:
            raise RuntimeError("No reference objects were accumulated")
        per_threshold = {}
        for threshold in IOU_THRESHOLDS:
            per_threshold[f"{threshold:.2f}"] = float(np.mean(
                [self._average_precision(class_id, float(threshold)) for class_id in class_ids]))
        return {
            "map50": per_threshold["0.50"],
            "map5095": float(np.mean(list(per_threshold.values()))),
            "evaluated_images": len(self.image_ids),
            "evaluated_classes": len(class_ids),
            "map_by_iou": per_threshold,
        }


def masks_from_boxes(boxes: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """Rasterise xyxy boxes into rectangular masks, for the self-check."""
    masks = torch.zeros((len(boxes), height, width), dtype=torch.bool)
    clamped = boxes.clone()
    clamped[:, 0::2] = clamped[:, 0::2].clamp(0, width)
    clamped[:, 1::2] = clamped[:, 1::2].clamp(0, height)
    for index, (x0, y0, x1, y1) in enumerate(clamped.round().long().tolist()):
        masks[index, y0:y1, x0:x1] = True
    return masks


def self_check() -> None:
    """MaskMAP on rectangular masks must equal DetectionMAP on the boxes that made them."""
    from dcvc_rt.src.utils.detection_map import DetectionMAP

    generator = torch.Generator().manual_seed(11)
    height = width = 128
    box_metric, mask_metric = DetectionMAP(), MaskMAP()
    for image_id in range(6):
        target_count = int(torch.randint(2, 5, (1,), generator=generator))
        top_left = torch.randint(0, 70, (target_count, 2), generator=generator).float()
        size = torch.randint(20, 48, (target_count, 2), generator=generator).float()
        target_boxes = torch.cat((top_left, top_left + size), dim=1)
        target_classes = torch.randint(0, 3, (target_count,), generator=generator)

        # Most predictions are jittered copies of the targets, so the AP sweep lands
        # strictly between 0 and 1 and the two metrics are compared on real matches.
        jitter = torch.randint(-7, 8, target_boxes.shape, generator=generator).float()
        predicted_boxes = torch.cat((target_boxes + jitter, torch.tensor([[2.0, 2.0, 14.0, 14.0]])))
        # Keep every box inside the raster so box area and mask area describe one shape.
        predicted_boxes[:, 0::2] = predicted_boxes[:, 0::2].clamp(0, width)
        predicted_boxes[:, 1::2] = predicted_boxes[:, 1::2].clamp(0, height)
        predicted_classes = torch.cat((target_classes,
                                       torch.randint(0, 3, (1,), generator=generator)))
        scores = torch.rand(len(predicted_boxes), generator=generator)

        box_metric.add(image_id=image_id, predicted_boxes=predicted_boxes,
                       predicted_scores=scores, predicted_classes=predicted_classes,
                       target_boxes=target_boxes, target_classes=target_classes)
        mask_metric.add(image_id=image_id,
                        predicted_masks=masks_from_boxes(predicted_boxes, height, width),
                        predicted_scores=scores, predicted_classes=predicted_classes,
                        target_masks=masks_from_boxes(target_boxes, height, width),
                        target_classes=target_classes)

    box_result, mask_result = box_metric.compute(), mask_metric.compute()
    for key in ("map50", "map5095", "evaluated_images", "evaluated_classes"):
        assert abs(box_result[key] - mask_result[key]) < 1e-6, (key, box_result[key],
                                                                mask_result[key])

    # A reference compared with itself must score a perfect 1.0.
    perfect = MaskMAP()
    masks = masks_from_boxes(torch.tensor([[10.0, 10.0, 50.0, 50.0], [60.0, 60.0, 90.0, 100.0]]),
                             height, width)
    classes = torch.tensor([0, 1])
    perfect.add(image_id=0, predicted_masks=masks, predicted_scores=torch.tensor([0.9, 0.8]),
                predicted_classes=classes, target_masks=masks, target_classes=classes)
    result = perfect.compute()
    assert abs(result["map50"] - 1.0) < 1e-9 and abs(result["map5095"] - 1.0) < 1e-9, result

    # Disjoint masks must score 0.
    empty = MaskMAP()
    other = masks_from_boxes(torch.tensor([[0.0, 0.0, 5.0, 5.0]]), height, width)
    empty.add(image_id=0, predicted_masks=other, predicted_scores=torch.tensor([0.9]),
              predicted_classes=torch.tensor([0]), target_masks=masks[:1],
              target_classes=torch.tensor([0]))
    assert empty.compute()["map50"] == 0.0

    print("mask_map self-check passed: rectangular masks reproduce DetectionMAP exactly "
          f"(map50 {box_result['map50']:.6f}, map5095 {box_result['map5095']:.6f}), "
          "identical masks score 1.0, disjoint masks score 0.0")


if __name__ == "__main__":
    self_check()
