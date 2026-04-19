from __future__ import annotations

from dataclasses import dataclass


REPO_URL = "https://github.com/chenguolin/DiffSplat.git"
T3BENCH_URL = "https://raw.githubusercontent.com/3DTopia/T3Bench/main/data/t3bench_prompt.txt"
GSO_REPO_ID = "ashawkey/gso_part0_rendered"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


@dataclass(frozen=True)
class ModelSpec:
    model_type: str
    script: str
    config: str
    text_tag: str
    image_tag: str
    no_render_tag: str


MODEL_SPECS: dict[str, ModelSpec] = {
    "sd15": ModelSpec(
        model_type="sd15",
        script="src/infer_gsdiff_sd.py",
        config="configs/gsdiff_sd15.yaml",
        text_tag="gsdiff_gobj83k_sd15__render",
        image_tag="gsdiff_gobj83k_sd15_image__render",
        no_render_tag="gsdiff_gobj83k_sd15",
    ),
    "pas": ModelSpec(
        model_type="pas",
        script="src/infer_gsdiff_pas.py",
        config="configs/gsdiff_pas.yaml",
        text_tag="gsdiff_gobj83k_pas_fp16__render",
        image_tag="gsdiff_gobj83k_pas_fp16_image__render",
        no_render_tag="gsdiff_gobj83k_pas_fp16",
    ),
    "sd35m": ModelSpec(
        model_type="sd35m",
        script="src/infer_gsdiff_sd3.py",
        config="configs/gsdiff_sd35m_80g.yaml",
        text_tag="gsdiff_gobj83k_sd35m__render",
        image_tag="gsdiff_gobj83k_sd35m_image__render",
        no_render_tag="gsdiff_gobj83k_sd35m",
    ),
}

