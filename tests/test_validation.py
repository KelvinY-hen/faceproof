from __future__ import annotations

import cv2
import numpy as np

import main
from conftest import jpg, png


def post_photo(client, data: bytes, content_type: str = "image/png", **params):
    return client.post(
        "/validate-applicant-photo",
        params=params,
        files={"applicant_photo": ("photo.png", data, content_type)},
    )


def test_health_reports_both_models(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] and body["recognition_model_loaded"]


def test_index_page_renders(client):
    assert "Faceproof" in client.get("/").text


def test_formal_photo_passes(client, id_photo):
    body = post_photo(client, png(id_photo)).json()
    assert body["valid"], body["errors"]
    assert body["metrics"]["face_count"] == 1
    assert body["metrics"]["background"]["type"] == "white"
    assert "embedding" not in body


def test_blue_background_passes(client, id_photo):
    blue = id_photo.copy()
    blue[(id_photo == 255).all(axis=2)] = (200, 120, 40)  # BGR
    body = post_photo(client, jpg(blue, 95), "image/jpeg").json()
    assert body["valid"], body["errors"]
    assert body["metrics"]["background"]["type"] == "blue"


def test_can_return_embedding_with_validation(client, id_photo):
    body = post_photo(client, png(id_photo), include_embedding="true").json()
    assert body["valid"]
    assert body["embedding_model"] == "sface_2021dec"
    assert len(body["embedding"]) == 684  # base64 of 128 float32


def test_coloured_background_fails(client, id_photo):
    grey = id_photo.copy()
    grey[(id_photo == 255).all(axis=2)] = (120, 140, 150)
    assert post_photo(client, png(grey)).json()["errors"] == ["Photo background must be plain white or blue."]


def test_landscape_photo_fails(client, id_photo):
    wide = cv2.copyMakeBorder(id_photo, 0, 0, 150, 150, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    errors = post_photo(client, png(wide)).json()["errors"]
    assert any(e.startswith("Photo must be portrait size") for e in errors)


def test_tilted_head_fails(client, id_photo):
    rotation = cv2.getRotationMatrix2D((150, 180), 15, 1.0)
    tilted = cv2.warpAffine(id_photo, rotation, (300, 400), borderValue=(255, 255, 255))
    assert "Applicant face must not be tilted." in post_photo(client, png(tilted)).json()["errors"]


def test_blurry_photo_fails(client, id_photo):
    blurry = cv2.GaussianBlur(id_photo, (0, 0), 6)
    assert "Applicant photo is too blurry." in post_photo(client, png(blurry)).json()["errors"]


def test_no_face(client):
    blank = np.full((400, 300, 3), 255, np.uint8)
    assert "No face detected in applicant photo." in post_photo(client, png(blank)).json()["errors"]


def test_two_faces(client, id_photo):
    pair = np.hstack([id_photo, id_photo])
    errors = post_photo(client, png(pair)).json()["errors"]
    assert "Only one face is allowed in applicant photo." in errors


def test_rejects_other_file_types(client):
    response = post_photo(client, b"GIF89a", "image/gif")
    assert response.status_code == 400
    assert response.json()["valid"] is False


def test_rejects_unreadable_image(client):
    assert post_photo(client, b"not an image").json()["errors"] == ["Unable to read uploaded photo."]


def test_rejects_large_upload(client):
    response = post_photo(client, b"\0" * (main.MAX_UPLOAD_BYTES + 1))
    assert response.status_code == 400
    assert "too large" in response.json()["errors"][0]


def test_large_photo_is_detected_after_downscale(client, id_photo):
    big = cv2.resize(id_photo, (2400, 3200), interpolation=cv2.INTER_CUBIC)
    body = post_photo(client, jpg(big, 85), "image/jpeg").json()
    assert body["metrics"]["face_count"] == 1
    # landmarks are mapped back to full-size coordinates
    assert body["metrics"]["face"]["width"] > 1000


def test_frontal_check_follows_the_eye_line():
    left, right = (0.0, 0.0), (100.0, 20.0)  # head tilted about 11 degrees
    mid = (50.0, 10.0)
    below_mid = (50.0 - 8.0, 10.0 + 40.0)  # straight down the face, perpendicular to the eye line
    assert main.sideways_offset(below_mid, mid, left, right, float(np.hypot(100, 20))) < 0.01
