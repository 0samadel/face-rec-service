# face-rec-service/app.py (FINAL - Corrected Indentation and Logic)

from flask import Flask, request, jsonify
from deepface import DeepFace
from deepface.commons import distance as dst
import numpy as np
import base64
from io import BytesIO
from PIL import Image
import json
import traceback

app = Flask(__name__)

MODEL_NAME = "SFace"
DETECTOR_BACKEND = "retinaface"

# Pre-load the model on startup
try:
    print("Loading face recognition model...")
    _ = DeepFace.represent(
        img_path=np.zeros((100, 100, 3)),
        model_name=MODEL_NAME,
        detector_backend=DETECTOR_BACKEND
    )
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model on startup: {e}")


def b64_to_numpy(b64_string):
    if "data:image" in b64_string:
        b64_string = b64_string.split(',')[1]
    img_bytes = base64.b64decode(b64_string)
    pil_img = Image.open(BytesIO(img_bytes)).convert("RGB")
    return np.array(pil_img)


@app.route("/generate-embedding", methods=["POST"])
def generate_embedding():
    try:
        if "face" in request.files:
            up_file = request.files["face"]
            img_bytes = up_file.read()
            np_img = np.array(Image.open(BytesIO(img_bytes)).convert("RGB"))
        elif request.json and request.json.get("image_base64"):
            np_img = b64_to_numpy(request.json["image_base64"])
        else:
            return jsonify(error="No image provided (expected 'face' file or 'image_base64' JSON field)."), 400

        embedding_objs = DeepFace.represent(
            img_path=np_img,
            model_name=MODEL_NAME,
            enforce_detection=True,
            detector_backend=DETECTOR_BACKEND
        )
        
        embedding = embedding_objs[0]['embedding']
        return jsonify(embedding=embedding), 200

    except ValueError as e:
        return jsonify(error=f"No face detected in the image. Error: {e}"), 400
    except Exception as e:
        traceback.print_exc()
        return jsonify(error=f"An internal error occurred: {e}"), 500


@app.route("/compare-faces", methods=["POST"])
def compare_faces():
    try:
        if "face" not in request.files:
            return jsonify(error="No face image provided for comparison."), 400

        stored_embedding_json = request.form.get("stored_embedding")
        if not stored_embedding_json:
            return jsonify(error="No stored embedding provided."), 400
        
        up_file = request.files["face"]
        img_bytes = up_file.read()
        np_img = np.array(Image.open(BytesIO(img_bytes)).convert("RGB"))
        
        unknown_embedding_objs = DeepFace.represent(
            img_path=np_img,
            model_name=MODEL_NAME,
            enforce_detection=True,
            detector_backend=DETECTOR_BACKEND
        )
        unknown_embedding = unknown_embedding_objs[0]['embedding']
        
        stored_embedding = np.array(json.loads(stored_embedding_json))
        
        distance_val = dst.findCosineDistance(unknown_embedding, stored_embedding)
        
        threshold = dst.getThreshold(MODEL_NAME, "cosine")
        is_match = distance_val <= threshold
        
        print(f"Comparison Result: Distance={distance_val:.4f}, Threshold={threshold}, Match={is_match}")
        
        return jsonify(is_match=is_match), 200

    except ValueError as e:
        return jsonify(error=f"Could not process image: {e}", is_match=False), 400
    except Exception as e:
        traceback.print_exc()
        return jsonify(error=f"An internal error occurred during comparison: {e}", is_match=False), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
