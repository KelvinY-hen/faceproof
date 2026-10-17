from __future__ import annotations

import numpy as np

from conftest import jpg, png, portrait


def verify(client, applicant: bytes, reference: bytes | None = None, embedding: str | None = None):
    files = {"applicant_photo": ("a.jpg", applicant, "image/jpeg")}
    if reference is not None:
        files["reference_photo"] = ("r.jpg", reference, "image/jpeg")
    data = {"reference_embedding": embedding} if embedding is not None else {}
    return client.post("/verify-face", files=files, data=data)


def enrol(client, photo: bytes) -> str:
    body = client.post("/face-embedding", files={"photo": ("p.jpg", photo, "image/jpeg")}).json()
    assert body["valid"], body["errors"]
    return body["embedding"]


def test_same_person_different_photos_match(client):
    body = verify(client, jpg(portrait("obama_2012")), jpg(portrait("obama_2009"))).json()
    assert body["match"] is True
    assert body["similarity"] > body["threshold"]


def test_same_person_years_apart_match(client):
    assert verify(client, jpg(portrait("biden_2021")), jpg(portrait("biden_2013"))).json()["match"] is True


def test_different_people_do_not_match(client):
    body = verify(client, jpg(portrait("obama_2012")), jpg(portrait("biden_2021"))).json()
    assert body["match"] is False
    assert body["similarity"] < body["threshold"]


def test_cropped_id_photo_matches_full_portrait(client, id_photo):
    assert verify(client, png(id_photo), jpg(portrait("obama_2012"))).json()["match"] is True


def test_verify_against_stored_embedding(client):
    stored = enrol(client, jpg(portrait("biden_2013")))
    assert verify(client, jpg(portrait("biden_2021")), embedding=stored).json()["match"] is True
    assert verify(client, jpg(portrait("obama_2009")), embedding=stored).json()["match"] is False


def test_embedding_endpoint_needs_a_face(client):
    blank = np.full((300, 300, 3), 255, np.uint8)
    body = client.post("/face-embedding", files={"photo": ("p.png", png(blank), "image/png")}).json()
    assert body == {"valid": False, "errors": ["No face detected in photo."], "metrics": {}}


def test_no_face_in_reference(client):
    blank = np.full((300, 300, 3), 255, np.uint8)
    body = verify(client, jpg(portrait("obama_2009")), jpg(blank)).json()
    assert body["match"] is False
    assert body["errors"] == ["No face detected in reference photo."]


def test_group_photo_is_ambiguous(client, id_photo):
    pair = np.hstack([id_photo, id_photo])
    body = verify(client, png(pair), jpg(portrait("obama_2012"))).json()
    assert body["errors"] == ["More than one face found in applicant photo."]


def test_reference_is_required(client):
    assert verify(client, jpg(portrait("obama_2009"))).status_code == 400


def test_only_one_reference_allowed(client):
    photo = jpg(portrait("obama_2009"))
    stored = enrol(client, photo)
    assert verify(client, photo, photo, stored).status_code == 400


def test_malformed_embedding(client):
    response = verify(client, jpg(portrait("obama_2009")), embedding="bm90IGFuIGVtYmVkZGluZw==")
    assert response.status_code == 400
