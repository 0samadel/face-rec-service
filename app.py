# face-rec-service/app.py (Updated to use DeepFace)

from flask import Flask, request, jsonify
from deepface import DeepFace
import numpy as np
import base64
from io import BytesIO
from PIL import Image
import json

app = Flask(__name__)

# Define the model and detector backend we will use consistently.
# SFace is lightweight and highly accurate. 'retinaface' is a robust detector.
MODEL_NAME = "SFace"
DETECTOR_BACKEND = "retinaface"

# --------------------------------------------------
# Pre-load the model on startup to make the first API call fast.
# We create a dummy image to pass to the model.
# --------------------------------------------------
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

# --------------------------------------------------
# Helper: Convert base64 string to a NumPy image array
# --------------------------------------------------
def b64_to_numpy(b64_string):
    if "data:image" in b64_string:
        b64_string = b64_string.split(',')[1]
    img_bytes = base64.b64decode(b64_string)
    pil_img = Image.open(BytesIO(img_bytes)).convert("RGB")
    return np.array(pil_img)

# --------------------------------------------------
# Main route for generating a face embedding
# Accepts either a multipart file or a base64 string.
# --------------------------------------------------
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

        # Use DeepFace to represent the face as a vector (embedding)
        # 'enforce_detection=True' ensures it fails if no face is found.
        embedding_objs = DeepFace.represent(
            img_path=np_img,
            model_name=MODEL_NAME,
            enforce_detection=True,
            detector_backend=DETECTOR_BACKEND
        )

        # DeepFace returns a list of objects, one for each detected face. We take the first.
        embedding = embedding_objs[0]['embedding']
        return jsonify(embedding=embedding), 200

    except ValueError as e:
        # This specific error is thrown by DeepFace when no face is detected
        return jsonify(error=f"No face detected in the image. Error: {e}"), 400
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return jsonify(error=f"An internal error occurred: {e}"), 500

# --------------------------------------------------
# Route for comparing a new face to a stored embedding
# --------------------------------------------------
@app.route("/compare-faces", methods=["POST"])
def compare_faces():
    try:
        if "face" not in request.files:
            return jsonify(error="No face image provided for comparison."), 400

        stored_embedding_json = request.form.get("stored_embedding")
        if not stored_embedding_json:
            return jsonify(error="No stored embedding provided."), 400

        # Convert the stored JSON embedding back to a numpy array
        stored_embedding = np.array(json.loads(stored_embedding_json))

        # Read the uploaded image
        up_file = request.files["face"]
        img_bytes = up_file.read()
        np_img = np.array(Image.open(BytesIO(img_bytes)).convert("RGB"))

        # Use DeepFace.verify() which is optimized for this task
        result = DeepFace.verify(
            img1_path=np_img,
            img2_path=stored_embedding, # DeepFace can accept an embedding directly
            model_name=MODEL_NAME,
            enforce_detection=True,
            detector_backend=DETECTOR_BACKEND
        )

        return jsonify(is_match=result["verified"]), 200

    except ValueError as e:
        # This could happen if no face is detected in the uploaded image
        return jsonify(error=f"Could not process image. Error: {e}", is_match=False), 400
    except Exception as e:
        print(f"An unexpected error occurred during comparison: {e}")
        return jsonify(error=f"An internal error occurred: {e}", is_match=False), 500


# --------------------------------------------------
# Entry-point for running the Flask application
# --------------------------------------------------
if __name__ == "__main__":
    # For local development
    app.run(host="0.0.0.0", port=5001, debug=True)
