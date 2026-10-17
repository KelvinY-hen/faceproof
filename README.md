# Faceproof: ID-photo checks and face matching API

A small FastAPI service that:

1. **Checks an ID or passport-style photo** before a form is submitted: one face, centred, the right size, not tilted or turned, sharp, and on a plain white or blue background.
2. **Recognises faces.** It turns a face into a 128-number embedding and checks whether two photos show the same person (1:1 verification), for example a selfie against an ID card or against a face enrolled earlier.

It comes with a browser demo page that draws the detected face and landmarks and explains every rule's result.

| Purpose | Model (OpenCV Zoo) | OpenCV class |
|---|---|---|
| Face detection + 5 landmarks | YuNet `face_detection_yunet_2023mar.onnx` (230 KB) | `cv2.FaceDetectorYN` |
| Face recognition (128-d embedding) | SFace `face_recognition_sface_2021dec.onnx` (37 MB) | `cv2.FaceRecognizerSF` |

**Stack:** Python 3.11+ · FastAPI · OpenCV 4.10 (DNN) · NumPy · pytest · Docker

## Run

```bash
python -m venv .venv
.venv\Scripts\activate                 # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --port 8090
```

- http://127.0.0.1:8090/ is the demo page.
- http://127.0.0.1:8090/docs is the Swagger UI.

To run it with Docker, use `docker compose up -d --build`. The API is then on port **8089**.

### Tests

```bash
pip install -r requirements-dev.txt
pytest
```

There are 27 tests. On the first run they download four public-domain official US government portraits from Wikimedia Commons into `tests/.cache/`, and they skip themselves when offline. The tests:
- build a passport-style photo from those portraits
- break it on purpose (grey background, tilt, blur, two faces, landscape crop) to check each rule
- confirm that different photos of the same person match and that different people don't

## API

### `POST /validate-applicant-photo`

- Multipart field `applicant_photo`: a JPEG or PNG, up to 10 MB.
- Add `?include_embedding=true` to also get the face embedding, so you can validate and enrol in one call.

```json
{
  "valid": false,
  "errors": ["Photo background must be plain white or blue."],
  "metrics": {"face_count": 1, "face": {...}, "landmarks": {...}, "pose": {...}, "blur_score": 812.4, "background": {...}}
}
```

### `POST /face-embedding`

Multipart field `photo`. Returns the embedding of the largest face as base64 of 128 little-endian float32 values (684 characters). Store it with the person's record.

### `POST /verify-face`

Multipart field `applicant_photo` plus exactly one of these:
- `reference_photo`: an image file
- `reference_embedding`: a value returned by `/face-embedding`

```json
{"match": true, "similarity": 0.7563, "threshold": 0.363, "errors": [], "metrics": {"applicant": {...}, "reference": {...}}}
```

- `similarity` is the cosine similarity of the two embeddings.
- The default threshold, 0.363, is OpenCV's reference value. Override it with `FACE_MATCH_THRESHOLD`.
- In the test photos, the same person scores 0.75–0.80 and different people score 0.3 or less.
- A second face smaller than 25% of the main face is ignored, such as the small ghost portrait on an ID card. A larger second face makes the photo ambiguous.

### Status codes

- `200`: processed. Read `valid`/`match` and `errors`.
- `400`: bad request, for example the wrong file type, a file that's too large, a missing or duplicate reference, or a malformed embedding.
- `503`: a model didn't load. See `GET /health`.

## Validation rules

| Rule | Default |
|---|---|
| Exactly one face | YuNet score ≥ 0.85 |
| Portrait photo | width / height 0.55–0.90 |
| Face size | ≥ 10% of the image area; face height 25–70% of the image height |
| Centred | ≤ 12% horizontal / 20% vertical offset |
| Not tilted | eye line within 8° |
| Facing the camera | nose and mouth near the eye midpoint, measured along the eye line |
| Sharp | Laplacian variance of the face ≥ 15 |
| Plain background | ≥ 80% of the pixels outside the head and shoulders are white or blue |

The thresholds are constants at the top of `main.py`.

## Design notes

- **The detector is shared and guarded by a lock.** YuNet keeps state (input size, score threshold), so a single instance is protected with a lock. The models load once at startup.
- **Handlers are plain `def`** rather than `async`, so FastAPI runs the CPU-heavy OpenCV work in its thread pool and one slow photo doesn't block other requests.
- **Large phone photos are shrunk for detection only.** The detected landmarks are mapped back to the full image before alignment and embedding.
- **Uploads are capped while reading.** The file is never fully buffered past 10 MB.

## Limits

- This is **not liveness detection**. A printed photo or a phone screen held up to the camera can pass, so add a liveness step for live identity checks.
- There's no mask or obstruction check. A covered face is only caught when YuNet can't find it.
- Embeddings are biometric data. Store them as personal data and delete them along with the person's record.
