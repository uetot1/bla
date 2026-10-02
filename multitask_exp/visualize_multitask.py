"""Qualitative figures: one decoded frame, two tasks.

Chapter 4 says one bitstream serves detection and instance segmentation; this script shows
it. For a chosen SFU frame it codes the whole sequence with each codec, decodes it, and
runs the frozen YOLOv5s (boxes) and YOLOv5s-seg (masks) on the SAME decoded frame.

* ``compare``: rows = uncompressed / HEVC / each neural codec at one rate point, columns =
  detection | segmentation, every panel titled with the sequence bpp of that rate point.
* ``ladder``: one codec over every base QP, so both tasks are seen degrading together.

Nothing here feeds a number in the thesis. Bits come from the audited coding paths used
for the RD curves (``evaluate_vcm.encode_sequence``; ``evaluate_hevc``'s x265 path), so the
bpp in a title is the bpp of that sequence at that rate point. The display confidence
(0.25 for both models) is for readability only; the box mAP axis scores at 0.001.

Experimental (multitask_exp). The notebook drives the functions in process; the CLI
covers the CPU self-check:
    python -m multitask_exp.visualize_multitask --self_check
"""

import argparse
import dataclasses
import json
import shutil
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from multitask_exp.kitti_mots import letterbox_geometry, letterbox_masks

DISPLAY_CONFIDENCE = 0.25
# Tableau-like colours; one per class id modulo the list, so a class keeps its colour
# across every panel of a figure.
PALETTE = np.array([
    (0.12, 0.47, 0.71), (1.00, 0.50, 0.05), (0.17, 0.63, 0.17), (0.84, 0.15, 0.16),
    (0.58, 0.40, 0.74), (0.55, 0.34, 0.29), (0.89, 0.47, 0.76), (0.50, 0.50, 0.50),
    (0.74, 0.74, 0.13), (0.09, 0.75, 0.81),
])


def colour(index):
    return PALETTE[int(index) % len(PALETTE)]


def safe(value):
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in str(value))


def limit_frames(sequence, count):
    """The sequence cut to its first `count` frames (None keeps it whole)."""
    if count is None or count >= sequence.frame_count:
        return sequence
    if count < 2:
        raise ValueError("Need an I-frame and at least one P-frame")
    return dataclasses.replace(sequence, frame_paths=sequence.frame_paths[:count],
                               label_paths=sequence.label_paths[:count])


def masks_to_source(masks, height, width, size):
    """[n,size,size] letterbox masks -> [n,height,width] bool numpy in source pixels."""
    new_width, new_height, top, _, left, _ = letterbox_geometry(height, width, size)
    if not len(masks):
        return np.zeros((0, height, width), dtype=bool)
    cropped = masks[:, top:top + new_height, left:left + new_width].float()[:, None]
    resized = torch.nn.functional.interpolate(cropped, size=(height, width), mode="bilinear",
                                              align_corners=False)
    return (resized[:, 0] > 0.5).cpu().numpy()


class BoxPredictor:
    """Frozen YOLOv5s through AutoShape, exactly as the box axis loads it."""

    def __init__(self, weights, device, size=640, confidence=DISPLAY_CONFIDENCE,
                 nms_iou=0.6, max_detections=300):
        from dcvc_rt.src.models.yolov5_extractor import load_yolov5

        self.model = load_yolov5("yolov5s", weights=weights).to(device).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.model.conf, self.model.iou, self.model.max_det = confidence, nms_iou, max_detections
        self.size = int(size)
        self.names = self.model.names

    @torch.inference_mode()
    def __call__(self, image):
        """HWC uint8 RGB -> dict(boxes [n,4] xyxy source pixels, scores, classes)."""
        rows = self.model([image], size=self.size).xyxy[0].detach().cpu().numpy()
        return {"boxes": rows[:, :4], "scores": rows[:, 4], "classes": rows[:, 5].astype(int)}


class MaskPredictor:
    """Frozen YOLOv5s-seg (the mask axis predictor) with masks mapped back to source pixels."""

    def __init__(self, weights, device, size=640, confidence=DISPLAY_CONFIDENCE,
                 nms_iou=0.45, max_detections=300):
        from multitask_exp.evaluate_mask_proxy import SegPredictor

        self.predictor = SegPredictor(weights, device, size, confidence, nms_iou, max_detections)
        self.size = int(size)
        self.names = self.predictor.names

    def __call__(self, image):
        """HWC uint8 RGB -> dict(masks [n,H,W] bool, scores, classes)."""
        masks, scores, classes = self.predictor(image)
        height, width = image.shape[:2]
        return {"masks": masks_to_source(masks, height, width, self.size),
                "scores": scores.cpu().numpy(), "classes": classes.cpu().numpy().astype(int)}


def choose_frame(dataset, sequence, mask_predictor, step=10, skip_first=True):
    """Index of the frame with most instances on the uncompressed source, P-frames only."""
    from multitask_exp.evaluate_mask_proxy import to_uint8_rgb

    best, best_count = None, -1
    for index in range(1 if skip_first else 0, sequence.frame_count, step):
        image = to_uint8_rgb(dataset.load_frame(sequence.frame_paths[index]))
        count = len(mask_predictor(image)["classes"])
        if count > best_count:
            best, best_count = index, count
    return best, best_count


class NeuralCodec:
    """DMCI loaded once; DMC checkpoints swapped per codec. fp16 like every thesis table."""

    def __init__(self, image_ckpt, device, precision="fp16", force_zero_thres=0.12,
                 reset_interval=32):
        from dcvc_rt.src.models.image_model import DMCI
        from evaluate_vcm import load_codec_checkpoint

        self.device, self.precision = device, precision
        self.force_zero_thres, self.reset_interval = force_zero_thres, reset_interval
        self.image_model = DMCI().to(device).eval()
        load_codec_checkpoint(self.image_model, image_ckpt, state_key="dmci_state_dict")
        self.image_model.update(force_zero_thres=force_zero_thres)
        if precision == "fp16":
            self.image_model.half()

    def load(self, video_ckpt):
        from dcvc_rt.src.models.video_model import DMC
        from evaluate_vcm import load_codec_checkpoint

        model = DMC().to(self.device).eval()
        checkpoint = load_codec_checkpoint(model, video_ckpt)
        model.update(force_zero_thres=self.force_zero_thres)
        if self.precision == "fp16":
            model.half()
        return model, bool(checkpoint.get("hierarchical_qp", True))

    @torch.inference_mode()
    def frames(self, model, hierarchical_qp, dataset, sequence, base_qp, wanted, work_dir):
        """Code the sequence, decode up to max(wanted); ({index: uint8 RGB}, rate dict)."""
        from dcvc_rt.src.utils.transforms import ycbcr2rgb
        from dcvc_rt.src.utils.vcm_bitstream import VCMSequenceReader
        from evaluate_vcm import container_bits_by_frame, encode_sequence
        from multitask_exp.evaluate_mask_proxy import to_uint8_rgb

        path = Path(work_dir) / f"{safe(sequence.name)}_q{base_qp:02d}.vcm"
        path.parent.mkdir(parents=True, exist_ok=True)
        encode_sequence(self.image_model, model, dataset, sequence, base_qp, path,
                        self.device, self.reset_interval, hierarchical_qp)
        rate = rate_record(sequence, path.stat().st_size * 8, container_bits_by_frame(path))
        wanted, decoded_frames = set(wanted), {}
        model.clear_dpb()
        model.set_curr_poc(0)
        with VCMSequenceReader(path) as reader:
            header = reader.header
            sps = {"height": sequence.height, "width": sequence.width,
                   "ec_part": int(header.two_entropy_coders), "use_ada_i": 0}
            for index, packet in enumerate(reader.frames()):
                if index == 0:
                    decoded = self.image_model.decompress(packet.bitstream, sps, packet.qp)
                    model.add_ref_frame(feature=None, frame=decoded["x_hat"])
                else:
                    if header.reset_interval > 0 and index % header.reset_interval == 1:
                        model.reset_ref_feature()
                    decoded = model.decompress(packet.bitstream, sps, packet.qp)
                if index in wanted:
                    rgb = ycbcr2rgb(decoded["x_hat"][:, :, : sequence.height, : sequence.width])
                    decoded_frames[index] = to_uint8_rgb(rgb.float())
                if index >= max(wanted):
                    break
        path.unlink()
        return decoded_frames, rate


def rate_record(sequence, actual_bits, bits_by_frame=None):
    pixels = sequence.width * sequence.height
    record = {"actual_bits": int(actual_bits),
              "bpp": actual_bits / (sequence.frame_count * pixels),
              "frames": sequence.frame_count}
    if bits_by_frame is not None:
        record["frame_bpp"] = [bits / pixels for bits in bits_by_frame]
    return record


class HevcCodec:
    """x265 through evaluate_hevc's functions: the same anchor as every BD-rate."""

    def __init__(self, work_dir, x265="x265", ffmpeg="ffmpeg", preset="medium",
                 bit_depth=8, chroma_format="420"):
        from evaluate_hevc import resolve_executable

        self.work_dir = Path(work_dir)
        self.x265 = resolve_executable(x265, "x265")
        self.ffmpeg = resolve_executable(ffmpeg, "FFmpeg")
        self.preset, self.bit_depth, self.chroma_format = preset, bit_depth, chroma_format

    def source_yuv(self, sequence):
        from evaluate_hevc import rgb_to_yuv, write_concat_file

        folder = self.work_dir / safe(sequence.name)
        folder.mkdir(parents=True, exist_ok=True)
        yuv = folder / f"source_{sequence.frame_count}.yuv"
        if not yuv.is_file():
            write_concat_file(sequence, folder / "frames.ffconcat")
            rgb_to_yuv(self.ffmpeg, sequence, folder / "frames.ffconcat", yuv,
                       self.bit_depth, self.chroma_format)
        return yuv

    def frames(self, sequence, qp, wanted):
        from evaluate_hevc import x265_decode, x265_encode, yuv_to_rgb_frames

        folder = self.work_dir / safe(sequence.name) / f"qp_{qp:02d}"
        folder.mkdir(parents=True, exist_ok=True)
        stream, decoded_yuv = folder / "stream.bin", folder / "decoded.yuv"
        x265_encode(self.x265, sequence, self.source_yuv(sequence), stream, qp,
                    self.bit_depth, self.chroma_format, self.preset, [])
        x265_decode(self.ffmpeg, sequence, stream, decoded_yuv, self.bit_depth, self.chroma_format)
        paths = yuv_to_rgb_frames(self.ffmpeg, sequence, decoded_yuv, folder / "png",
                                  self.bit_depth, self.chroma_format)
        images = {index: np.array(Image.open(paths[index]).convert("RGB")) for index in wanted}
        rate = rate_record(sequence, stream.stat().st_size * 8)
        shutil.rmtree(folder, ignore_errors=True)
        return images, rate

    def cleanup(self):
        shutil.rmtree(self.work_dir, ignore_errors=True)


# ----------------------------------------------------------------------------- drawing

def draw_boxes(ax, image, result, names):
    import matplotlib.patches as patches

    ax.imshow(image)
    for (x0, y0, x1, y1), score, cls in zip(result["boxes"], result["scores"], result["classes"]):
        ax.add_patch(patches.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False,
                                       edgecolor=colour(cls), linewidth=1.6))
        ax.text(x0, y0, f"{names[int(cls)]} {score:.2f}", fontsize=6, color="white",
                va="bottom", bbox={"facecolor": colour(cls), "edgecolor": "none", "pad": 1})
    ax.set_axis_off()


def draw_masks(ax, image, result, names, alpha=0.5):
    overlay = image.astype(np.float32) / 255.0
    for mask, cls in zip(result["masks"], result["classes"]):
        overlay[mask] = (1 - alpha) * overlay[mask] + alpha * colour(cls)
    ax.imshow(overlay.clip(0, 1))
    for mask, cls in zip(result["masks"], result["classes"]):
        if mask.any():
            ax.contour(mask.astype(float), levels=[0.5], colors=[colour(cls)], linewidths=0.8)
            ys, xs = np.nonzero(mask)
            ax.text(xs.min(), ys.min(), names[int(cls)], fontsize=6, color="white", va="bottom",
                    bbox={"facecolor": colour(cls), "edgecolor": "none", "pad": 1})
    ax.set_axis_off()


def panel_title(row, kind, result):
    count = len(result["classes"])
    unit = "hộp" if kind == "box" else "mask"
    rate = "không nén" if row.get("bpp") is None else f"{row['bpp']:.3f} bpp"
    return f"{row['label']} · {rate} · {count} {unit}"


def render_grid(rows, names, path, title, column_titles=("Detection · YOLOv5s",
                                                       "Instance segmentation · YOLOv5s-seg")):
    """rows: list of dict(label, bpp, image, box, mask), drawn as len(rows) x 2."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    height, width = rows[0]["image"].shape[:2]
    panel_width = 5.2
    panel_height = panel_width * height / width + 0.35
    figure, axes = plt.subplots(len(rows), 2, figsize=(2 * panel_width, len(rows) * panel_height),
                                squeeze=False)
    for r, row in enumerate(rows):
        draw_boxes(axes[r][0], row["image"], row["box"], names["box"])
        draw_masks(axes[r][1], row["image"], row["mask"], names["mask"])
        axes[r][0].set_title(panel_title(row, "box", row["box"]), fontsize=9)
        axes[r][1].set_title(panel_title(row, "mask", row["mask"]), fontsize=9)
    for c, text in enumerate(column_titles):
        axes[0][c].annotate(text, (0.5, 1.0), xycoords="axes fraction", xytext=(0, 22),
                            textcoords="offset points", ha="center", fontsize=11,
                            fontweight="bold")
    figure.suptitle(title, fontsize=12, y=1.0)
    figure.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)
    return path


def render_ladder(rows, names, path, title):
    """rows ordered by bpp: two figure rows (boxes, masks), one column per rate point."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    height, width = rows[0]["image"].shape[:2]
    panel_width = 3.4
    panel_height = panel_width * height / width + 0.35
    figure, axes = plt.subplots(2, len(rows), figsize=(len(rows) * panel_width, 2 * panel_height),
                                squeeze=False)
    for c, row in enumerate(rows):
        draw_boxes(axes[0][c], row["image"], row["box"], names["box"])
        draw_masks(axes[1][c], row["image"], row["mask"], names["mask"])
        rate = "không nén" if row.get("bpp") is None else f"{row['bpp']:.3f} bpp"
        axes[0][c].set_title(f"{row['label']} · {rate}\n{len(row['box']['classes'])} hộp",
                             fontsize=8)
        axes[1][c].set_title(f"{len(row['mask']['classes'])} mask", fontsize=8)
    axes[0][0].annotate("Detection", (-0.04, 0.5), xycoords="axes fraction", rotation=90,
                        ha="right", va="center", fontsize=10, fontweight="bold")
    axes[1][0].annotate("Segmentation", (-0.04, 0.5), xycoords="axes fraction", rotation=90,
                        ha="right", va="center", fontsize=10, fontweight="bold")
    figure.suptitle(title, fontsize=11, y=1.0)
    figure.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)
    return path


def predict_row(label, bpp, image, box_predictor, mask_predictor):
    return {"label": label, "bpp": bpp, "image": image,
            "box": box_predictor(image), "mask": mask_predictor(image),
            "names": {"box": box_predictor.names, "mask": mask_predictor.names}}


def summary_row(row):
    """JSON-safe description of a panel: rate and what each model found."""
    def classes(result, names):
        found = {}
        for cls in result["classes"]:
            found[names[int(cls)]] = found.get(names[int(cls)], 0) + 1
        return found
    return {"label": row["label"], "bpp": row["bpp"],
            "boxes": int(len(row["box"]["classes"])), "masks": int(len(row["mask"]["classes"])),
            "box_classes": classes(row["box"], row["names"]["box"]),
            "mask_classes": classes(row["mask"], row["names"]["mask"])}


# ----------------------------------------------------------------------------- self-check

def self_check(args):
    """CPU only: geometry round trip, frame limiting, both renderers. No codec, no YOLO."""
    import tempfile

    # A source-space mask sent into letterbox space (the human-label path) and back here
    # must land on the same pixels, for the resolutions this project draws.
    for height, width in ((240, 416), (480, 832), (374, 1242)):
        mask = torch.zeros((1, height, width), dtype=torch.bool)
        mask[0, height // 4: height // 2, width // 3: width // 2] = True
        back = masks_to_source(letterbox_masks(mask, 640), height, width, 640)[0]
        original = mask[0].numpy()
        iou = (back & original).sum() / (back | original).sum()
        assert back.shape == (height, width) and iou > 0.95, (height, width, iou)

    @dataclasses.dataclass(frozen=True)
    class FakeSequence:
        name: str
        frame_paths: tuple
        label_paths: tuple
        width: int = 416
        height: int = 240

        @property
        def frame_count(self):
            return len(self.frame_paths)

    sequence = FakeSequence("s", tuple(range(10)), tuple(range(10)))
    assert limit_frames(sequence, None) is sequence and limit_frames(sequence, 99) is sequence
    assert limit_frames(sequence, 4).frame_count == 4
    try:
        limit_frames(sequence, 1)
    except ValueError:
        pass
    else:
        raise AssertionError("one frame must be refused")
    rate = rate_record(sequence, 416 * 240 * 10 * 0.5, [416 * 240 * 0.5] * 10)
    assert abs(rate["bpp"] - 0.5) < 1e-12 and len(rate["frame_bpp"]) == 10

    generator = np.random.default_rng(3)
    image = (generator.random((240, 416, 3)) * 255).astype(np.uint8)
    masks = np.zeros((2, 240, 416), dtype=bool)
    masks[0, 40:120, 50:150] = True
    masks[1, 100:200, 250:380] = True
    result_box = {"boxes": np.array([[50, 40, 150, 120], [250, 100, 380, 200]], float),
                  "scores": np.array([0.9, 0.6]), "classes": np.array([0, 2])}
    result_mask = {"masks": masks, "scores": np.array([0.9, 0.6]), "classes": np.array([0, 2])}
    names = {"box": {0: "person", 2: "car"}, "mask": {0: "person", 2: "car"}}
    rows = [{"label": label, "bpp": bpp, "image": image, "box": result_box, "mask": result_mask,
             "names": names} for label, bpp in (("Gốc", None), ("R4f q63", 0.21))]
    with tempfile.TemporaryDirectory() as folder:
        grid = render_grid(rows, names, Path(folder) / "grid.png", "self-check")
        ladder = render_ladder(rows, names, Path(folder) / "ladder.png", "self-check")
        assert grid.stat().st_size > 10_000 and ladder.stat().st_size > 10_000
    assert summary_row(rows[1])["mask_classes"] == {"person": 1, "car": 1}
    json.dumps([summary_row(row) for row in rows])
    print("visualize_multitask self-check passed: letterbox round trip IoU > 0.95 at three "
          "resolutions, frame limiting, rate record, both renderers")


def parse_args():
    parser = argparse.ArgumentParser(description="Qualitative two-task figures")
    parser.add_argument("--self_check", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    if arguments.self_check:
        self_check(arguments)
    else:
        raise SystemExit("Run from the notebook kaggle_multitask_visualize.ipynb, "
                         "or --self_check")
