from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

from diffsplat_tools.common import apply_hf_env, require_repo, run
from diffsplat_tools.constants import GSO_REPO_ID, MODEL_SPECS, T3BENCH_URL


def download_checkpoints(
    repo_dir: Path,
    out_dir: Path,
    *,
    model: str,
    variant: str,
    env: dict[str, str],
    dry_run: bool,
) -> None:
    require_repo(repo_dir, dry_run=dry_run)
    out_dir.mkdir(parents=True, exist_ok=True)
    spec = MODEL_SPECS[model]
    base = [sys.executable, "download_ckpt.py", "--local_dir", str(out_dir), "--model_type", spec.model_type]

    if variant in {"text", "both"}:
        run(base, cwd=repo_dir, env=env, dry_run=dry_run)
    if variant in {"image", "both"}:
        run([*base, "--image_cond"], cwd=repo_dir, env=env, dry_run=dry_run)


def download_t3bench(data_dir: Path, dry_run: bool) -> None:
    target = data_dir / "t3bench" / "t3bench_prompt.txt"
    if target.exists():
        print(f"T3Bench prompts already exist at {target}")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading T3Bench prompts to {target}")
    if not dry_run:
        urllib.request.urlretrieve(T3BENCH_URL, target)


def download_gso(
    data_dir: Path,
    dry_run: bool,
    *,
    hf_home: str | None = None,
    hf_endpoint: str | None = None,
) -> None:
    target = data_dir / "gso_rendered"
    if target.exists() and any(target.iterdir()):
        print(f"GSO rendered dataset already exists at {target}")
        return
    print(f"Downloading GSO rendered dataset to {target}")
    if dry_run:
        return
    apply_hf_env(hf_home=hf_home, hf_endpoint=hf_endpoint)
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise SystemExit(
            "huggingface_hub is required for GSO download. Run install first or `pip install huggingface_hub`."
        ) from exc
    snapshot_download(repo_id=GSO_REPO_ID, repo_type="dataset", local_dir=str(target))
