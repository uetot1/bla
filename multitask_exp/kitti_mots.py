"""KITTI-MOTS: human-annotated instance masks for the segmentation axis.

Every mask number on SFU is scored against a proxy reference -- what the frozen
yolov5s-seg predicts on the uncompressed frame -- because SFU has no mask labels. This
module lets the same evaluators score against masks drawn by people instead.

KITTI-MOTS (Voigtlaender et al., CVPR 2019) annotates cars and pedestrians in 21 KITTI
tracking training sequences. Each frame has one 16-bit PNG whose pixel value encodes the
object: ``class_id * 1000 + instance_id``; class 1 is car, 2 is pedestrian, and the value
10000 marks an "ignore" region (objects too small or ambiguous to annotate).

What this module does, and nothing else:

* ``prepare_dataset`` turns the KITTI images plus the instance PNGs into the project's
  ``AnnotatedVideoDataset`` layout (frames/, labels/ with YOLO boxes taken from the masks,
  masks/ with the instance PNGs, manifest.json). Frames are cropped to even width and
  height -- x265 4:2:0 needs that (``evaluate_hevc.py`` refuses odd sizes) -- and the
  masks are cropped identically, so every codec sees the same pixels.
* ``HumanMaskReference`` loads a frame's masks and moves them into the detector's
  letterbox space, the same space ``SegPredictor`` returns its masks in, using exactly the
  geometry of yolov5's ``letterbox``.
* ``drop_ignored`` removes predictions that lie mostly inside an ignore region, so a
  codec is not charged for finding an object the annotators chose not to label.

Class ids are mapped to COCO, the label space of both YOLO models: car -> 2, pedestrian ->
0 (person).
"""

import argparse
import json
import math
import shutil
from pathlib import Path

import numpy as np
import torch
from PIL import Image

# The standard KITTI-MOTS validation split (TrackR-CNN, Voigtlaender et al. 2019).
VAL_SEQUENCES = ("0002", "0006", "0007", "0008", "0010", "0013", "0014", "0016", "0018")
KITTI_TO_COCO = {1: 2, 2: 0}          # car -> car, pedestrian -> person
IGNORE_VALUE = 10000
FPS = 10.0                            # KITTI camera rate
IGNORE_FRACTION = 0.5                 # a prediction this much inside "ignore" is dropped
GROUND_TRUTH = ("KITTI-MOTS human-annotated instance masks (car, pedestrian), validation "
                "split; predictions mostly inside annotated ignore regions are discarded")


def decode_instances(id_map):
    """16-bit KITTI-MOTS id map [H,W] -> (masks [n,H,W] bool, coco classes [n], ignore [H,W])."""
    id_map = np.asarray(id_map).astype(np.int64)
    ignore = id_map == IGNORE_VALUE
    masks, classes = [], []
    for value in np.unique(id_map):
        if value == 0 or value == IGNORE_VALUE:
            continue
        kitti_class = int(value) // 1000
        if kitti_class not in KITTI_TO_COCO:
            # Not a class KITTI-MOTS evaluates; treat like an ignore region.
            ignore |= id_map == value
            continue
        masks.append(id_map == value)
        classes.append(KITTI_TO_COCO[kitti_class])
    height, width = id_map.shape
    stacked = np.stack(masks) if masks else np.zeros((0, height, width), dtype=bool)
    return stacked, np.asarray(classes, dtype=np.int64), ignore


def read_id_map(path):
    with Image.open(path) as image:
        return np.asarray(image, dtype=np.int64)


def even_size(width, height):
    return width - width % 2, height - height % 2


def yolo_rows(masks, classes, width, height):
    """Tight box of each mask, as normalized YOLO rows (class xc yc w h)."""
    rows = []
    for mask, class_id in zip(masks, classes):
        ys, xs = np.nonzero(mask)
        if len(xs) == 0:
            continue
        left, right = xs.min(), xs.max() + 1
        top, bottom = ys.min(), ys.max() + 1
        rows.append(f"{int(class_id)} {(left + right) / 2 / width:.6f} "
                    f"{(top + bottom) / 2 / height:.6f} {(right - left) / width:.6f} "
                    f"{(bottom - top) / height:.6f}")
    return rows


def find_roots(search_root):
    """Locate (images root with <seq>/*.png, instances root with <seq>/*.png) under a tree.

    Images: KITTI tracking ``.../training/image_02/<seq>/``. Instances: a folder whose
    children are sequence folders of 16-bit PNGs, normally named ``instances``. A zip of
    the instances is extracted next to the working copy first.
    """
    search_root = Path(search_root)
    images = [p.parent for p in search_root.rglob("image_02/0002")
              if p.is_dir() and p.parent.parent.name == "training"]
    instances = [p.parent for p in search_root.rglob("0002")
                 if p.is_dir() and p.parent.name == "instances"
                 and any(p.glob("*.png"))]
    return images, instances


def prepare_dataset(images_root, instances_root, out_dir, sequences=VAL_SEQUENCES,
                    max_frames=None):
    """Write the AnnotatedVideoDataset layout. Returns (data_dir, manifest_path, stats)."""
    images_root, instances_root, out_dir = Path(images_root), Path(instances_root), Path(out_dir)
    entries, stats = [], {"frames": 0, "instances": {"car": 0, "person": 0},
                          "frames_without_label_png": 0, "cropped_from": {}}
    for name in sequences:
        image_dir, label_png_dir = images_root / name, instances_root / name
        if not image_dir.is_dir():
            raise FileNotFoundError(f"KITTI images for sequence {name} not found: {image_dir}")
        if not label_png_dir.is_dir():
            raise FileNotFoundError(f"KITTI-MOTS masks for sequence {name} not found: {label_png_dir}")
        frame_paths = sorted(image_dir.glob("*.png"))
        if max_frames:
            frame_paths = frame_paths[:max_frames]
        frames_out = out_dir / "frames" / name
        labels_out = out_dir / "labels" / name
        masks_out = out_dir / "masks" / name
        for folder in (frames_out, labels_out, masks_out):
            folder.mkdir(parents=True, exist_ok=True)
        for frame_path in frame_paths:
            with Image.open(frame_path) as image:
                rgb = image.convert("RGB")
                width, height = even_size(*rgb.size)
                if (width, height) != rgb.size:
                    stats["cropped_from"][name] = f"{rgb.size[0]}x{rgb.size[1]} -> {width}x{height}"
                rgb.crop((0, 0, width, height)).save(frames_out / frame_path.name)
            label_png = label_png_dir / frame_path.name
            if label_png.is_file():
                id_map = read_id_map(label_png)[:height, :width]
            else:
                stats["frames_without_label_png"] += 1
                id_map = np.zeros((height, width), dtype=np.int64)
            if id_map.shape != (height, width):
                raise ValueError(f"{label_png}: mask size {id_map.shape} does not match "
                                 f"frame {height}x{width}")
            Image.fromarray(id_map.astype(np.uint16)).save(masks_out / frame_path.name)
            masks, classes, _ = decode_instances(id_map)
            (labels_out / f"{frame_path.stem}.txt").write_text(
                "\n".join(yolo_rows(masks, classes, width, height)), encoding="utf-8")
            stats["frames"] += 1
            stats["instances"]["car"] += int((classes == 2).sum())
            stats["instances"]["person"] += int((classes == 0).sum())
        entries.append({"name": f"KITTI_{name}", "frames_dir": f"frames/{name}",
                        "labels_dir": f"labels/{name}", "fps": FPS})
    manifest = out_dir / "manifest.json"
    manifest.write_text(json.dumps({"sequences": entries}, indent=2), encoding="utf-8")
    return out_dir, manifest, stats


def letterbox_geometry(height, width, size):
    """Exactly yolov5 utils.augmentations.letterbox with auto=False, scaleup=True."""
    ratio = min(size / height, size / width)
    new_width, new_height = int(round(width * ratio)), int(round(height * ratio))
    pad_w, pad_h = (size - new_width) / 2, (size - new_height) / 2
    top, bottom = int(round(pad_h - 0.1)), int(round(pad_h + 0.1))
    left, right = int(round(pad_w - 0.1)), int(round(pad_w + 0.1))
    return new_width, new_height, top, bottom, left, right


def letterbox_masks(masks, size):
    """[n,H,W] bool tensor -> [n,size,size] bool, nearest resize + zero padding."""
    count, height, width = masks.shape
    new_width, new_height, top, bottom, left, right = letterbox_geometry(height, width, size)
    if count == 0:
        return torch.zeros((0, size, size), dtype=torch.bool, device=masks.device)
    resized = masks[None].float()
    if (new_height, new_width) != (height, width):
        resized = torch.nn.functional.interpolate(resized, size=(new_height, new_width),
                                                  mode="nearest")
    padded = torch.nn.functional.pad(resized, (left, right, top, bottom))
    return padded[0] > 0.5


class HumanMaskReference:
    """Ground truth for one frame, in the detector's letterbox space."""

    def __init__(self, masks_dir, size, device):
        self.masks_dir = Path(masks_dir)
        self.size = int(size)
        self.device = device

    def __call__(self, sequence, frame_index):
        folder = sequence.name.removeprefix("KITTI_")
        path = self.masks_dir / folder / sequence.frame_paths[frame_index].name
        masks, classes, ignore = decode_instances(read_id_map(path))
        stacked = torch.from_numpy(np.concatenate([masks, ignore[None]])).to(self.device)
        boxed = letterbox_masks(stacked, self.size)
        return boxed[:-1], torch.from_numpy(classes).to(self.device), boxed[-1]


def drop_ignored(masks, scores, classes, ignore, fraction=IGNORE_FRACTION):
    """Remove predictions with at least `fraction` of their area inside the ignore region."""
    if ignore is None or masks.shape[0] == 0 or not bool(ignore.any()):
        return masks, scores, classes
    area = masks.flatten(1).sum(1).float()
    inside = (masks & ignore[None]).flatten(1).sum(1).float()
    keep = inside < fraction * area.clamp_min(1)
    return masks[keep], scores[keep], classes[keep]


def self_check(args):
    import tempfile

    from dcvc_rt.src.utils.vcm_eval_dataset import AnnotatedVideoDataset
    from multitask_exp.mask_map import MaskMAP

    # 1. Decoding the id encoding.
    id_map = np.zeros((13, 21), dtype=np.int64)
    id_map[2:6, 3:8] = 1001          # car
    id_map[7:12, 10:14] = 2003       # pedestrian
    id_map[0:2, 15:21] = IGNORE_VALUE
    masks, classes, ignore = decode_instances(id_map)
    assert masks.shape == (2, 13, 21) and classes.tolist() == [2, 0], (masks.shape, classes)
    assert masks[0].sum() == 20 and masks[1].sum() == 20 and ignore.sum() == 12

    # 2. prepare_dataset on a synthetic KITTI tree with an odd height, read back by the
    #    project's own dataset class; boxes must be the masks' tight boxes.
    with tempfile.TemporaryDirectory() as folder:
        folder = Path(folder)
        images = folder / "kitti" / "training" / "image_02" / "0002"
        instances = folder / "labels" / "instances" / "0002"
        images.mkdir(parents=True)
        instances.mkdir(parents=True)
        rng = np.random.default_rng(0)
        for index in range(3):
            Image.fromarray(rng.integers(0, 255, (13, 21, 3), dtype=np.uint8)).save(
                images / f"{index:06d}.png")
            Image.fromarray(id_map.astype(np.uint16)).save(instances / f"{index:06d}.png")
        found_images, found_instances = find_roots(folder)
        assert len(found_images) == 1 and len(found_instances) == 1, (found_images, found_instances)
        data_dir, manifest, stats = prepare_dataset(found_images[0], found_instances[0],
                                                    folder / "prepared", sequences=("0002",))
        assert stats["frames"] == 3 and stats["instances"] == {"car": 3, "person": 3}, stats
        assert stats["cropped_from"] == {"0002": "21x13 -> 20x12"}, stats
        dataset = AnnotatedVideoDataset(data_dir, manifest)
        sequence = next(iter(dataset))
        assert (sequence.width, sequence.height) == (20, 12)
        boxes, box_classes = dataset.load_ground_truth(sequence.label_paths[0], 20, 12)
        assert box_classes.tolist() == [2, 0]
        assert torch.allclose(boxes, torch.tensor([[3., 2., 8., 6.], [10., 7., 14., 12.]]),
                              atol=1e-3), boxes

        # 3. The reference in letterbox space: scored against itself it must read 1.0.
        reference = HumanMaskReference(data_dir / "masks", size=64, device=torch.device("cpu"))
        target_masks, target_classes, target_ignore = reference(sequence, 0)
        assert target_masks.shape == (2, 64, 64) and target_ignore.shape == (64, 64)
        metric = MaskMAP()
        metric.add(image_id=0, predicted_masks=target_masks,
                   predicted_scores=torch.tensor([0.9, 0.8]), predicted_classes=target_classes,
                   target_masks=target_masks, target_classes=target_classes)
        assert abs(metric.compute()["map50"] - 1.0) < 1e-9

    # 4. Letterbox geometry against yolov5's own letterbox, on KITTI's real frame sizes.
    geometry_note = "yolov5 not importable here, geometry compared to formula only"
    hub = Path(torch.hub.get_dir()) / "ultralytics_yolov5_v7.0"
    if hub.is_dir():
        import sys
        sys.path.insert(0, str(hub))
        from utils.augmentations import letterbox
        for height, width in ((374, 1242), (370, 1224), (374, 1238), (376, 1240), (480, 640)):
            image = np.full((height, width, 3), 255, dtype=np.uint8)
            theirs = letterbox(image, new_shape=640, stride=32, auto=False)[0][..., 0] == 255
            ours = letterbox_masks(torch.ones((1, height, width), dtype=torch.bool), 640)[0]
            assert torch.equal(torch.from_numpy(theirs), ours), (height, width)
        geometry_note = "letterbox matches yolov5 pixel for pixel on 5 KITTI/SFU sizes"

    # 5. Ignore handling.
    ignore = torch.zeros((10, 10), dtype=torch.bool)
    ignore[:, :5] = True
    inside = torch.zeros((1, 10, 10), dtype=torch.bool)
    inside[0, 2:4, 1:4] = True
    straddle = torch.zeros((1, 10, 10), dtype=torch.bool)
    straddle[0, 2:4, 4:8] = True           # 1/4 inside -> kept
    predicted = torch.cat([inside, straddle])
    kept, scores, kept_classes = drop_ignored(predicted, torch.tensor([0.9, 0.8]),
                                              torch.tensor([2, 0]), ignore)
    assert kept.shape[0] == 1 and kept_classes.tolist() == [0], kept_classes

    print(f"kitti_mots self-check passed: id decoding, dataset preparation with even crop "
          f"and tight boxes, reference identity 1.0, ignore filter; {geometry_note}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--self_check", action="store_true")
    arguments = parser.parse_args()
    if arguments.self_check:
        self_check(arguments)
