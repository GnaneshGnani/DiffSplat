#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MEAN_RE = re.compile(
    r"Mean\s+CosSim:\s+(?P<clipsim>[-+.\w]+)\s+R-Prec:\s+(?P<rprec>[-+.\w]+)\s+ImageReward:\s+(?P<reward>[-+.\w]+)"
)


def parse_float(value: str):
    try:
        return float(value)
    except ValueError:
        return None


def parse_metrics(log_path: Path) -> dict[str, float | None]:
    metrics = {"clipsim": None, "rprec": None, "imagereward": None}
    if not log_path.exists():
        return metrics
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = MEAN_RE.search(line)
        if match:
            metrics = {
                "clipsim": parse_float(match.group("clipsim")),
                "rprec": parse_float(match.group("rprec")),
                "imagereward": parse_float(match.group("reward")),
            }
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect DiffSplat ablation manifests and log metrics.")
    parser.add_argument("--root", default="out_ablation")
    parser.add_argument("--json", default="results/ablation_summary.json")
    parser.add_argument("--csv", default="results/ablation_summary.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = (ROOT / args.root).resolve()
    rows: list[dict[str, object]] = []
    for manifest_path in sorted(root.glob("*/manifest.json")):
        exp_dir = manifest_path.parent
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        inference_dir = exp_dir / "inference"
        row: dict[str, object] = dict(manifest)
        row.pop("command", None)
        row.update(parse_metrics(exp_dir / "log_infer.txt"))
        row["num_gs_png"] = len(list(inference_dir.glob("*_gs.png")))
        row["num_gif"] = len(list(inference_dir.glob("*.gif")))
        row["num_mp4"] = len(list(inference_dir.glob("*.mp4")))
        row["num_ply"] = len(list(inference_dir.glob("*.ply")))
        row["exp_dir"] = str(exp_dir)
        rows.append(row)

    json_path = (ROOT / args.json).resolve()
    csv_path = (ROOT / args.csv).resolve()
    json_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    if rows:
        fieldnames = sorted({key for row in rows for key in row})
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    else:
        csv_path.write_text("", encoding="utf-8")
    print(f"Wrote {len(rows)} rows to {json_path}")
    print(f"Wrote CSV to {csv_path}")


if __name__ == "__main__":
    main()
