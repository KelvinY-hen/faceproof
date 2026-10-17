"""Face fixtures are public-domain official White House portraits from Wikimedia
Commons, downloaded once into tests/.cache (not committed). Tests skip offline."""
from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CACHE = Path(__file__).resolve().parent / ".cache"
COMMONS = "https://commons.wikimedia.org/wiki/Special:FilePath/"
PORTRAITS = {
    "obama_2009": "Official_portrait_of_Barack_Obama.jpg",
    "obama_2012": "President_Barack_Obama.jpg",
    "biden_2013": "Joe_Biden_official_portrait_2013.jpg",
    "biden_2021": "Joe_Biden_presidential_portrait.jpg",
}


def portrait(key: str) -> np.ndarray:
    path = CACHE / f"{key}.jpg"
    if not path.exists():
        CACHE.mkdir(exist_ok=True)
        request = urllib.request.Request(
            COMMONS + PORTRAITS[key] + "?width=960",
            headers={"User-Agent": "applicant-photo-api-tests/1.0"},
        )
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                path.write_bytes(response.read())
        except OSError as exc:
            pytest.skip(f"cannot download test portrait {key}: {exc}")
    return cv2.imread(str(path))


def png(image: np.ndarray) -> bytes:
    return cv2.imencode(".png", image)[1].tobytes()


def jpg(image: np.ndarray, quality: int = 90) -> bytes:
    return cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])[1].tobytes()


def make_id_photo(image: np.ndarray) -> np.ndarray:
    """Head and shoulders cut out onto a plain white 300x400 canvas."""
    import main

    x, y, w, h = (int(v) for v in main.ENGINE.detect(image, 0.8)[0].box)
    cx, cy = x + w // 2, y + h // 2
    crop_h = int(h / 0.5)
    crop_w = int(crop_h * 0.75)
    crop = image[max(0, cy - int(crop_h * 0.45)):cy + int(crop_h * 0.55), cx - crop_w // 2:cx + crop_w // 2]
    crop = cv2.resize(crop, (300, int(300 * crop.shape[0] / crop.shape[1])))[:400]

    canvas = np.full((400, 300, 3), 255, np.uint8)
    canvas[400 - crop.shape[0]:] = crop
    mask = np.zeros((400, 300), np.uint8)
    cv2.ellipse(mask, (150, 180), (95, 135), 0, 0, 360, 255, -1)
    cv2.rectangle(mask, (50, 290), (250, 400), 255, -1)
    canvas[mask == 0] = 255
    return canvas


@pytest.fixture(scope="session")
def id_photo() -> np.ndarray:
    return make_id_photo(portrait("obama_2009"))


@pytest.fixture(scope="session")
def client():
    from fastapi.testclient import TestClient

    import main

    return TestClient(main.app)
