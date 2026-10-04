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

from __future__ import annotations

import io
import json
import subprocess
import sys
from pathlib import Path

import httpx
import numpy as np
import pytest
from PIL import Image

from scripts import sam3_infer


@pytest.fixture
def cli_args(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> list[str]:
    checkpoint = tmp_path / "sam3.pt"
    checkpoint.touch()
    monkeypatch.setenv("SAM3_CHECKPOINT_PATH", str(checkpoint))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    Image.new("RGB", (80, 60), "white").save(tmp_path / "input.png")
    return [
        "--image",
        str(tmp_path / "input.png"),
        "--text",
        "红色杯子",
        "--output-dir",
        str(tmp_path / "result"),
    ]


def test_cli_preserves_all_instances_and_keeps_json_stdout_clean(
    cli_args: list[str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    masks = np.zeros((2, 60, 80), dtype=bool)
    masks[0, 10:20, 5:15] = True
    masks[1, 30:45, 40:60] = True

    def predict(image, text, checkpoint, min_score):
        assert image.size == (80, 60)
        assert text == "红色杯子"
        assert checkpoint == tmp_path / "sam3.pt"
        assert min_score == 0.2
        print("model progress")
        return (
            masks,
            np.array([[5.0, 10, 15, 20], [40, 30, 60, 45]]),
            np.array([0.4, 0.9]),
        )

    monkeypatch.setattr(sam3_infer, "_predict", predict)
    assert sam3_infer.main(cli_args) == 0
    captured = capsys.readouterr()
    result = json.loads(captured.out)
    assert "model progress" in captured.err
    assert result == json.loads((tmp_path / "result/result.json").read_text())
    assert result["found"] is True
    assert result["count"] == 2
    assert result["image_size"] == [80, 60]
    assert result["text"] == "红色杯子"
    assert result["detections"] == [
        {
            "id": 0,
            "score": 0.9,
            "box_xyxy": [40, 30, 60, 45],
            "mask_path": "masks/000.png",
        },
        {
            "id": 1,
            "score": 0.4,
            "box_xyxy": [5, 10, 15, 20],
            "mask_path": "masks/001.png",
        },
    ]
    for index, original in enumerate([masks[1], masks[0]]):
        with Image.open(
            tmp_path / "result" / result["detections"][index]["mask_path"]
        ) as saved:
            assert saved.mode == "L"
            np.testing.assert_array_equal(
                np.asarray(saved), original.astype(np.uint8) * 255
            )
    with Image.open(tmp_path / "result" / result["overlay_path"]) as overlay:
        assert overlay.size == (80, 60)
        assert overlay.getpixel((0, 59)) == (255, 255, 255)
        assert overlay.getpixel((50, 43)) != (255, 255, 255)


def test_no_detections_is_success_with_unmodified_overlay(
    cli_args: list[str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        sam3_infer,
        "_predict",
        lambda *args: (
            np.empty((0, 60, 80), dtype=bool),
            np.empty((0, 4)),
            np.empty(0),
        ),
    )
    assert sam3_infer.main(cli_args) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["found"] is False
    assert result["count"] == 0
    assert result["detections"] == []
    assert not (tmp_path / "result/masks").exists()
    with Image.open(tmp_path / "result/overlay.png") as overlay:
        np.testing.assert_array_equal(np.asarray(overlay), np.full((60, 80, 3), 255))


@pytest.mark.parametrize("status", [200, 404])
def test_http_image_follows_redirect_and_reports_http_errors(
    monkeypatch: pytest.MonkeyPatch,
    status: int,
) -> None:
    buffer = io.BytesIO()
    Image.new("RGBA", (7, 5), "red").save(buffer, format="PNG")

    def respond(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/image":
            return httpx.Response(302, headers={"location": "/actual.png"})
        return httpx.Response(status, content=buffer.getvalue())

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        monkeypatch.setattr(httpx, "get", client.get)
        if status == 404:
            with pytest.raises(httpx.HTTPStatusError):
                sam3_infer._load_image("https://example.test/image")
        else:
            image = sam3_infer._load_image("https://example.test/image")
            assert image.mode == "RGB"
            assert image.size == (7, 5)
            assert image.getpixel((0, 0)) == (255, 0, 0)


def test_image_uses_exif_orientation(tmp_path: Path) -> None:
    source = tmp_path / "rotated.jpg"
    exif = Image.Exif()
    exif[274] = 6
    Image.new("RGB", (7, 5), "red").save(source, exif=exif)
    assert sam3_infer._load_image(str(source)).size == (5, 7)


@pytest.mark.parametrize(
    "extra",
    [
        ["--min-score", "nan"],
        ["--min-score", "1.1"],
        ["--min-score", "-0.1"],
        ["--cuda-device", "-1"],
        ["--text", "  "],
    ],
)
def test_invalid_arguments_exit_before_inference(
    cli_args: list[str], extra: list[str]
) -> None:
    with pytest.raises(SystemExit) as exc:
        sam3_infer.main(cli_args + extra)
    assert exc.value.code == 2


@pytest.mark.parametrize("failure", ["checkpoint", "image", "inference", "output"])
def test_failed_run_emits_no_json_and_preserves_existing_results(
    cli_args: list[str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    failure: str,
) -> None:
    def predict(*args):
        raise RuntimeError("model failed")

    monkeypatch.setattr(sam3_infer, "_predict", predict)
    if failure == "checkpoint":
        (tmp_path / "sam3.pt").unlink()
    elif failure == "image":
        (tmp_path / "input.png").write_bytes(b"not an image")
    elif failure == "output":
        (tmp_path / "result").mkdir()
        (tmp_path / "result/result.json").write_text("previous result")
    assert sam3_infer.main(cli_args) == 1
    assert capsys.readouterr().out == ""
    if failure == "output":
        assert (tmp_path / "result/result.json").read_text() == "previous result"
    else:
        assert not (tmp_path / "result/result.json").exists()


def test_help_works_without_model_dependencies() -> None:
    script = Path(sam3_infer.__file__).resolve()
    probe = """
import builtins
import runpy
import sys
original_import = builtins.__import__
def checked_import(name, *args, **kwargs):
    if name.split('.')[0] in {'sam3', 'torch', 'PIL', 'numpy', 'httpx'}:
        raise ImportError('optional dependency unavailable')
    return original_import(name, *args, **kwargs)
builtins.__import__ = checked_import
sys.argv = [sys.argv[1], '--help']
runpy.run_path(sys.argv[0], run_name='__main__')
"""
    result = subprocess.run(
        [sys.executable, "-c", probe, str(script)],
        capture_output=True,
        text=True,
        check=False,
        timeout=15,
    )
    assert result.returncode == 0, result.stderr
    assert "--image" in result.stdout
    assert "--text" in result.stdout
