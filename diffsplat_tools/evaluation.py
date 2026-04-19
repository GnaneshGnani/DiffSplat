from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image

from diffsplat_tools.common import expand
from diffsplat_tools.constants import IMAGE_EXTENSIONS


def load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0


def psnr(pred: np.ndarray, gt: np.ndarray) -> float:
    mse = float(np.mean((pred - gt) ** 2))
    if mse <= 0:
        return float("inf")
    return -10.0 * math.log10(mse)


def collect_images(directory: Path) -> dict[str, Path]:
    images: dict[str, Path] = {}
    for path in directory.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            images[str(path.relative_to(directory))] = path
    return images


def match_images(pred_dir: Path, gt_dir: Path) -> list[tuple[Path, Path, str]]:
    pred = collect_images(pred_dir)
    gt = collect_images(gt_dir)
    shared = sorted(set(pred) & set(gt))
    if shared:
        return [(pred[key], gt[key], key) for key in shared]

    pred_by_name: dict[str, list[Path]] = {}
    gt_by_name: dict[str, list[Path]] = {}
    for key, value in pred.items():
        pred_by_name.setdefault(Path(key).name, []).append(value)
    for key, value in gt.items():
        gt_by_name.setdefault(Path(key).name, []).append(value)

    pairs: list[tuple[Path, Path, str]] = []
    for name in sorted(set(pred_by_name) & set(gt_by_name)):
        pred_paths = pred_by_name[name]
        gt_paths = gt_by_name[name]
        if len(pred_paths) == 1 and len(gt_paths) == 1:
            pairs.append((pred_paths[0], gt_paths[0], name))
    return pairs


def mean_and_std(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {"mean": float(array.mean()), "std": float(array.std())}


def evaluate_image_dirs(args) -> None:
    pred_dir = expand(args.pred_dir)
    gt_dir = expand(args.gt_dir)
    if not pred_dir.exists():
        raise SystemExit(f"Prediction directory not found: {pred_dir}")
    if not gt_dir.exists():
        raise SystemExit(f"Ground-truth directory not found: {gt_dir}")

    pairs = match_images(pred_dir, gt_dir)
    if not pairs:
        raise SystemExit("No matching images found between the prediction and ground-truth directories.")

    if args.limit is not None:
        pairs = pairs[: args.limit]

    device = args.device
    if device == "auto":
        try:
            import torch
        except ImportError:
            device = "cpu"
        else:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"

    lpips_model = None
    score_lpips = None
    if not args.skip_lpips:
        try:
            import lpips
            import torch
        except ImportError as exc:
            raise SystemExit("LPIPS evaluation requested, but `lpips` and `torch` are not installed.") from exc
        lpips_model = lpips.LPIPS(net="vgg").to(device).eval()

        def score_lpips(pred_arr: np.ndarray, gt_arr: np.ndarray) -> float:
            with torch.no_grad():
                pred_tensor = torch.from_numpy(pred_arr.transpose(2, 0, 1)).unsqueeze(0).float().to(device) * 2.0 - 1.0
                gt_tensor = torch.from_numpy(gt_arr.transpose(2, 0, 1)).unsqueeze(0).float().to(device) * 2.0 - 1.0
                return float(lpips_model(pred_tensor, gt_tensor).item())

    try:
        from skimage.metrics import structural_similarity
    except ImportError as exc:
        raise SystemExit(
            "scikit-image is required for SSIM. Run install first or `pip install scikit-image`."
        ) from exc

    rows: list[dict[str, float | str]] = []
    psnr_values: list[float] = []
    ssim_values: list[float] = []
    lpips_values: list[float] = []

    for pred_path, gt_path, key in pairs:
        pred = load_rgb(pred_path)
        gt = load_rgb(gt_path)
        if pred.shape != gt.shape:
            pred_image = Image.fromarray((pred * 255.0).astype(np.uint8))
            pred_image = pred_image.resize((gt.shape[1], gt.shape[0]), Image.Resampling.BICUBIC)
            pred = np.asarray(pred_image, dtype=np.float32) / 255.0

        pred_psnr = psnr(pred, gt)
        pred_ssim = float(structural_similarity(pred, gt, channel_axis=2, data_range=1.0))

        row: dict[str, float | str] = {"image": key, "psnr": pred_psnr, "ssim": pred_ssim}
        psnr_values.append(pred_psnr)
        ssim_values.append(pred_ssim)

        if score_lpips is not None:
            pred_lpips = score_lpips(pred, gt)
            row["lpips"] = pred_lpips
            lpips_values.append(pred_lpips)

        rows.append(row)

    summary: dict[str, object] = {
        "count": len(rows),
        "psnr": mean_and_std(psnr_values),
        "ssim": mean_and_std(ssim_values),
    }
    if lpips_values:
        summary["lpips"] = mean_and_std(lpips_values)
    if args.save_per_image:
        summary["per_image"] = rows

    print(json.dumps(summary, indent=2))
    if args.json:
        target = expand(args.json)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote metrics to {target}")

