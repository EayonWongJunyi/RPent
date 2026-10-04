# Copyright 2026 The RPent Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Segment one image with a text prompt using a local SAM3 checkpoint.

Requires RPent's ``sam3`` extra, a CUDA GPU, and local ``sam3.pt`` weights::

    conda activate rpent-py311
    export SAM3_CHECKPOINT_PATH=/path/to/sam3.pt
    python scripts/sam3_infer.py --image /path/to/image.jpg --text "red cup"

``--image`` also accepts an HTTP/HTTPS URL. Text is passed to SAM3 unchanged.
Each run writes result.json, overlay.png, and masks/000.png, masks/001.png, ...
in descending score order. Masks use 255 for foreground and 0 for background.
Sizes are [width, height]; boxes are pixel [left, top, right, bottom] coordinates
in the decoded, EXIF-oriented image. Artifact paths are relative to the output
directory, which must be empty or new. With no detections, the overlay is the
input image and the result has found=false, count=0, and detections=[].

Standard output contains only the result JSON; diagnostics go to standard
error. Exit codes are 0 for completed inference (including no detections),
1 for input/output or inference failures, and 2 for invalid CLI arguments.
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import os
import sys
from contextlib import redirect_stdout
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import urlsplit

from rpent.utils.logging import get_logger

if TYPE_CHECKING:
    import numpy as np
    from PIL import Image

logger = get_logger("sam3_infer")


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Segment an image with SAM3 and save every matching instance.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--image", required=True, help="Local image path or HTTP(S) URL."
    )
    parser.add_argument(
        "--text", required=True, help="Target description passed to SAM3."
    )
    parser.add_argument(
        "--checkpoint",
        default=os.environ.get("SAM3_CHECKPOINT_PATH"),
        help="Local sam3.pt path; defaults to SAM3_CHECKPOINT_PATH.",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=0.2,
        help="Keep all instances with scores above this threshold (0 to 1).",
    )
    parser.add_argument(
        "--cuda-device",
        type=int,
        default=0,
        help="GPU ordinal to expose through CUDA_VISIBLE_DEVICES.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="New or empty output directory; defaults to logs/sam3/<run-time>/.",
    )
    return parser


def _load_image(source: str) -> Image.Image:
    import httpx
    from PIL import Image, ImageOps

    if urlsplit(source).scheme.lower() in {"http", "https"}:
        response = httpx.get(source, follow_redirects=True, timeout=30.0)
        response.raise_for_status()
        data = response.content
    else:
        data = Path(source).expanduser().read_bytes()
    with Image.open(io.BytesIO(data)) as image:
        return ImageOps.exif_transpose(image).convert("RGB")


def _predict(
    image: Image.Image,
    text: str,
    checkpoint: Path,
    min_score: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    try:
        import torch
        from sam3.model.sam3_image_processor import Sam3Processor
        from sam3.model_builder import build_sam3_image_model
    except ImportError as exc:
        raise RuntimeError(
            'SAM3 dependencies are missing; install RPent with pip install -e ".[sam3]"'
        ) from exc

    if not torch.cuda.is_available():
        raise RuntimeError("SAM3 requires an available CUDA GPU; check --cuda-device")
    torch.cuda.set_device(0)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    logger.info("Loading SAM3 checkpoint: %s", checkpoint)
    model = build_sam3_image_model(
        device="cuda",
        checkpoint_path=str(checkpoint),
        load_from_HF=False,
        enable_inst_interactivity=False,
    )
    processor = Sam3Processor(model, device="cuda", confidence_threshold=min_score)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        state = processor.set_image(image)
        output = processor.set_text_prompt(prompt=text, state=state)
    return (
        output["masks"][:, 0].detach().cpu().numpy(),
        output["boxes"].detach().float().cpu().numpy(),
        output["scores"].detach().float().cpu().numpy(),
    )


def _save_result(
    image: Image.Image,
    masks: np.ndarray,
    boxes: np.ndarray,
    scores: np.ndarray,
    *,
    source: str,
    text: str,
    min_score: float,
    output_dir: Path,
) -> dict[str, Any]:
    import numpy as np
    from PIL import Image, ImageDraw

    palette = [
        (255, 80, 80),
        (65, 190, 255),
        (80, 220, 120),
        (255, 190, 60),
        (185, 100, 255),
        (40, 220, 210),
    ]
    pixels = np.array(image)
    detections = []
    if scores.size:
        (output_dir / "masks").mkdir()
    for instance_id, index in enumerate(np.argsort(-scores, kind="stable")):
        mask_path = f"masks/{instance_id:03d}.png"
        mask = masks[index]
        Image.fromarray(mask.astype(np.uint8) * 255).save(output_dir / mask_path)
        color = np.asarray(palette[instance_id % len(palette)])
        pixels[mask] = (0.55 * pixels[mask] + 0.45 * color).astype(np.uint8)
        detections.append(
            {
                "id": instance_id,
                "score": float(scores[index]),
                "box_xyxy": boxes[index].tolist(),
                "mask_path": mask_path,
            }
        )

    overlay = Image.fromarray(pixels)
    draw = ImageDraw.Draw(overlay)
    for detection in detections:
        color = palette[detection["id"] % len(palette)]
        box = detection["box_xyxy"]
        draw.rectangle(box, outline=color, width=2)
        label = f"{detection['id']}: {detection['score']:.3f}"
        bounds = draw.textbbox((0, 0), label)
        x = max(0, min(box[0], image.width - (bounds[2] - bounds[0]) - 4))
        y = max(0, min(box[1], image.height - (bounds[3] - bounds[1]) - 4))
        draw.rectangle(draw.textbbox((x, y), label), fill=color)
        draw.text((x, y), label, fill="black")
    overlay.save(output_dir / "overlay.png")
    result = {
        "image": source,
        "text": text,
        "image_size": list(image.size),
        "min_score": min_score,
        "found": bool(detections),
        "count": len(detections),
        "detections": detections,
        "overlay_path": "overlay.png",
    }
    (output_dir / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return result


def main(argv: list[str] | None = None) -> int:
    """Run one SAM3 inference and write its JSON result to standard output."""
    parser = _build_argparser()
    args = parser.parse_args(argv)
    if not args.text.strip():
        parser.error("--text must not be empty or whitespace")
    if not 0.0 <= args.min_score <= 1.0:
        parser.error("--min-score must be between 0 and 1")
    if args.cuda_device < 0:
        parser.error("--cuda-device must be nonnegative")
    if not args.checkpoint:
        parser.error("provide --checkpoint or set SAM3_CHECKPOINT_PATH")

    logging.basicConfig(level=logging.INFO, format="[%(name)s] %(message)s")
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    output_dir = (
        (
            args.output_dir
            or Path("logs/sam3") / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        )
        .expanduser()
        .resolve()
    )
    try:
        # Keep dependency initialization and model progress off the JSON stream.
        with redirect_stdout(sys.stderr):
            if not checkpoint.is_file():
                raise FileNotFoundError(f"SAM3 checkpoint not found: {checkpoint}")
            if output_dir.exists() and any(output_dir.iterdir()):
                raise ValueError(f"output directory must be empty or new: {output_dir}")
            image = _load_image(args.image)
            output_dir.mkdir(parents=True, exist_ok=True)
            target = str(args.cuda_device)
            previous = os.environ.get("CUDA_VISIBLE_DEVICES")
            if previous is not None and previous != target:
                logger.warning(
                    "Overriding CUDA_VISIBLE_DEVICES=%s with %s", previous, target
                )
            os.environ["CUDA_VISIBLE_DEVICES"] = target
            masks, boxes, scores = _predict(
                image, args.text, checkpoint, args.min_score
            )
            result = _save_result(
                image,
                masks,
                boxes,
                scores,
                source=args.image,
                text=args.text,
                min_score=args.min_score,
                output_dir=output_dir,
            )
            logger.info("Saved %d instances to %s", result["count"], output_dir)
    except Exception as exc:
        logger.error("SAM3 inference failed: %s", exc)
        return 1
    sys.stdout.write(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
