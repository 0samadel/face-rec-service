# face-rec-service/app.py
from flask import Flask, request, jsonify
import face_recognition
import numpy as np
import os, json, base64
from io import BytesIO
from PIL import Image

app = Flask(__name__)

UPLOAD_DIR = "temp_uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


# --------------------------------------------------
# Helper: convert base64 string → numpy image
# --------------------------------------------------
def b64_to_numpy(b64_string):
    if b64_string.startswith("data:image"):
        b64_string = b64_string.split(",")[1]  # strip data URI prefix
    img_bytes = base64.b64decode(b64_string)
    pil_img = Image.open(BytesIO(img_bytes)).convert("RGB")
    return np.array(pil_img)


# --------------------------------------------------
# Core logic – generate embedding from a numpy image
# --------------------------------------------------
def extract_embedding(np_image):
    face_locs = face_recognition.face_locations(np_image)
    if not face_locs:
        return None
    encodings = face_recognition.face_encodings(np_image, face_locs)
    return encodings[0].tolist()  # 128-d list


# --------------------------------------------------
# MAIN route: /generate-embedding  (file OR base64)
# --------------------------------------------------
@app.route("/generate-embedding", methods=["POST"])
def generate_embedding():
    try:
        # 1️⃣ Accept multipart file
        if "face" in request.files:
            up_file = request.files["face"]
            tmp_path = os.path.join(UPLOAD_DIR, up_file.filename)
            up_file.save(tmp_path)
            np_img = face_recognition.load_image_file(tmp_path)
            os.remove(tmp_path)

        # 2️⃣ Or accept base64 string in JSON/body form field
        elif request.json and request.json.get("image_base64"):
            np_img = b64_to_numpy(request.json["image_base64"])

        else:
            return jsonify(error="No face image provided (file or base64)."), 400

        embedding = extract_embedding(np_img)
        if not embedding:
            return jsonify(error="No face detected."), 400

        return jsonify(embedding=embedding), 200

    except Exception as e:
        return jsonify(error=str(e)), 500


# --------------------------------------------------
# Alias route: /enroll_face  -> re-use same view
# --------------------------------------------------
@app.route("/enroll_face", methods=["POST"])
def enroll_face_alias():
    return generate_embedding()


# --------------------------------------------------
# /compare-faces  (multipart file + stored_embedding)
# --------------------------------------------------
@app.route("/compare-faces", methods=["POST"])
def compare_faces():
    if "face" not in request.files:
        return jsonify(error="No face image for comparison."), 400

    stored_json = request.form.get("stored_embedding")
    if not stored_json:
        return jsonify(error="No stored embedding provided."), 400

    try:
        stored_emb = np.array(json.loads(stored_json), dtype="float64")
        up_file = request.files["face"]
        tmp_path = os.path.join(UPLOAD_DIR, up_file.filename)
        up_file.save(tmp_path)
        np_img = face_recognition.load_image_file(tmp_path)
        os.remove(tmp_path)

        unknown_embs = face_recognition.face_encodings(np_img)
        if not unknown_embs:
            return jsonify(error="No face found in uploaded image."), 400

        unknown_emb = unknown_embs[0]
        is_match = bool(face_recognition.compare_faces([stored_emb], unknown_emb, tolerance=0.5)[0])
        return jsonify(is_match=is_match), 200

    except Exception as e:
        return jsonify(error=str(e)), 500


# --------------------------------------------------
# Entry-point
# --------------------------------------------------
if __name__ == "__main__":
    # For production use:  gunicorn --bind 0.0.0.0:5001 app:app
    app.run(host="0.0.0.0", port=5001, debug=True)
