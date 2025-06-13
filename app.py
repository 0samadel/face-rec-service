# face-rec-service/app.py (FINAL - Most Robust Version)

from flask import Flask, request, jsonify
from deepface import DeepFace
import numpy as np
import base64
from io import BytesIO
from PIL import Image
import json
import traceback

app = Flask(__name__)

# --- Configuration ---
MODEL_NAME = "SFace"
DETECTOR_BACKEND = "retinaface"
MODEL = None # Global variable to hold the loaded model

# --- Function to load the model once ---
def load_model():
    global MODEL
    if MODEL is None:
        print("Model is not loaded. Loading now...")
        try:
            MODEL = DeepFace.build_model(MODEL_NAME)
            print("Model loaded successfully.")
        except Exception as e:
            print(f"FATAL: Could not load model on startup: {e}")
            MODEL = "ERROR" # Set a flag to indicate failure
    elif MODEL == "ERROR":
        print("Model previously failed to load. Aborting.")
    else:
        print("Model is already loaded.")

# Call the function once when the app starts.
load_model()

# --- Helper Function ---
def b64_to_numpy(b64_string):
    if "data:image" in b64_string:
        b64_string = b64_string.split(',')[1]
    img_bytes = base64.b64decode(b64_string)
    pil_img = Image.open(BytesIO(img_bytes)).convert("RGB")
    return np.array(pil_img)

# --- API Routes ---

@app.route("/generate-embedding", methods=["POST"])
def generate_embedding():
    if MODEL == "ERROR":
        return jsonify(error="Face recognition model is not available."), 503

    # ... (rest of the function is the same)
    try:
        if "face" in request.files:
            up_file = request.files["face"]
            img_bytes = up_file.read()
            np_img = np.array(Image.open(BytesIO(img_bytes)).convert("RGB"))
        else:
            return jsonify(error="No image file provided."), 400

        embedding_objs = DeepFace.represent(
            img_path=np_img,
            model_name=MODEL_NAME,
            enforce_detection=True,
            detector_backend=DETECTOR_BACKEND
        )
        embedding = embedding_objs[0]['embedding']
        return jsonify(embedding=embedding), 200
    except Exception as e:
        traceback.print_exc()
        return jsonify(error=f"Error in generate-embedding: {e}"), 500


@app.route("/compare-faces", methods=["POST"])
def compare_faces():
    if MODEL == "ERROR":
        return jsonify(error="Face recognition model is not available."), 503

    try:
        if "face" not in request.files:
            return jsonify(error="No face image provided for comparison."), 400
        stored_embedding_json = request.form.get("stored_embedding")
        if not stored_embedding_json:
            return jsonify(error="No stored embedding provided."), 400
        
        up_file = request.files["face"]
        img_bytes = up_file.read()
        np_img_to_verify = np.array(Image.open(BytesIO(img_bytes)).convert("RGB"))
        
        # We don't use DeepFace.verify because we want to compare against our own embedding
        # So we generate a new embedding and compare distances manually.
        
        new_embedding_objs = DeepFace.represent(
            img_path=np_img_to_verify,
            model_name=MODEL_NAME,
            enforce_detection=True,
            detector_backend=DETECTOR_BACKEND
        )
        new_embedding = np.array(new_embedding_objs[0]['embedding'])
        stored_embedding = np.array(json.loads(stored_embedding_json), dtype=np.float32)

        # Use the distance function from DeepFace's utility modules
        distance = np.linalg.norm(new_embedding - stored_embedding) # Euclidean distance is also common
        
        # Thresholds are model-specific. We need to find the correct one for SFace
        # For Cosine distance, it's ~0.593. For Euclidean L2, it's different.
        # Let's use cosine distance for consistency.
        from deepface.commons.distance import findCosineDistance
        cosine_distance = findCosineDistance(new_embedding, stored_embedding)
        threshold = 0.593 # Threshold for SFace with cosine distance
        
        is_match = cosine_distance <= threshold
        
        print(f"Comparison: Distance={cosine_distance:.4f}, Threshold={threshold}, Match={is_match}")
        
        return jsonify(is_match=is_match), 200

    except ValueError as e:
        return jsonify(error=f"Could not process image: {e}", is_match=False), 400
    except Exception as e:
        traceback.print_exc()
        return jsonify(error=f"An internal error occurred during comparison: {e}", is_match=False), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
