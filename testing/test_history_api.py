import json

from fastapi.testclient import TestClient
from PIL import Image
from PIL.PngImagePlugin import PngInfo

import backend.api.history as history_module
from backend.main import app


def test_history_lists_images_and_videos(tmp_path, monkeypatch):
    output_dir = tmp_path / "outputs"
    image_dir = output_dir / "batch_image123"
    video_dir = output_dir / "batch_video123"
    image_dir.mkdir(parents=True)
    video_dir.mkdir(parents=True)

    metadata = PngInfo()
    metadata.add_text("prompt", "a calm lake")
    metadata.add_text("batch_id", "image123")
    image_path = image_dir / "image123_1.png"
    Image.new("RGB", (4, 4), color=(12, 34, 56)).save(image_path, pnginfo=metadata)

    video_path = video_dir / "video123_1.mp4"
    video_path.write_bytes(b"fake mp4 bytes")
    sidecar_path = video_dir / "video_video123.mp4.json"
    sidecar_path.write_text(
        json.dumps(
            {
                "prompt": "waves rolling under moonlight",
                "negative_prompt": "jitter",
                "steps": 25,
                "cfg": 7.5,
                "batch_id": "video123",
                "videos": [
                    {
                        "filename": "video123_1.mp4",
                        "path": "batch_video123/video123_1.mp4",
                        "seed": 1234,
                        "index": 0,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    fallback_dir = output_dir / "batch_fallback456"
    fallback_dir.mkdir(parents=True)
    fallback_video_path = fallback_dir / "fallback456_1.mp4"
    fallback_video_path.write_bytes(b"fake fallback mp4 bytes")

    monkeypatch.setattr(history_module, "OUTPUT_DIR", output_dir)

    client = TestClient(app)
    response = client.get("/history")

    assert response.status_code == 200
    records = {record["filename"]: record for record in response.json()}

    image_record = records["batch_image123/image123_1.png"]
    assert image_record["url"] == "/outputs/batch_image123/image123_1.png"
    assert image_record["media_type"] == "image"
    assert image_record["metadata"]["prompt"] == "a calm lake"
    assert image_record["metadata"]["batch_id"] == "image123"

    video_record = records["batch_video123/video123_1.mp4"]
    assert video_record["url"] == "/outputs/batch_video123/video123_1.mp4"
    assert video_record["media_type"] == "video"
    assert video_record["metadata"]["batch_id"] == "video123"
    assert video_record["metadata"]["prompt"] == "waves rolling under moonlight"
    assert video_record["metadata"]["negative_prompt"] == "jitter"
    assert video_record["metadata"]["seed"] == 1234
    assert video_record["metadata"]["index"] == 0
    assert "videos" not in video_record["metadata"]

    fallback_record = records["batch_fallback456/fallback456_1.mp4"]
    assert fallback_record["media_type"] == "video"
    assert fallback_record["metadata"]["batch_id"] == "fallback456"
