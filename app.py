# face-rec-service/app.py (Definitive Final Version)

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
# For manual comparison, we use cosine. DeepFace.verify internally handles this.
DISTANCE_METRIC = "cosine" 

# --- Pre-load the model on startup ---
try:
    print("Loading face recognition model...")
    _ = DeepFace.build_model(MODEL_NAME)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model on startup: {e}")


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
    # This route works, no changes needed.
    try:
        if "face" in request.files:
            img_bytes = request.files["face"].read()
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
        return jsonify(error=f"An error occurred: {e}"), 500


@app.route("/compare-faces", methods=["POST"])
def compare_faces():
    try:
        if "face" not in request.files:
            return jsonify(error="No face image provided for comparison."), 400
        stored_embedding_json = request.form.get("stored_embedding")
        if not stored_embedding_json:
            return jsonify(error="No stored embedding provided."), 400
        
        # 1. Get the new image
        img_bytes = request.files["face"].read()
        np_img_to_verify = np.array(Image.open(BytesIO(img_bytes)).convert("RGB"))
        
        # 2. Get the stored embedding
        stored_embedding = json.loads(stored_embedding_json)

        # 3. ✅ Use the main DeepFace.verify function. This is the most stable method.
        # It handles all the complex logic internally.
        result = DeepFace.verify(
            img1_path=np_img_to_verify,
            img2_path=stored_embedding, # Pass the embedding directly to img2_path
            model_name=MODEL_NAME,
            detector_backend=DETECTOR_BACKEND,
            distance_metric=DISTANCE_METRIC,
            enforce_detection=True
        )
        
        is_match = result["verified"]
        
        print(f"Comparison Result: {result}")
        
        return jsonify(is_match=is_match), 200

    except ValueError as e:
        # This is often "Face could not be detected."
        return jsonify(error=f"Could not process image: {e}", is_match=False), 400
    except Exception as e:
        traceback.print_exc()
        return jsonify(error=f"An internal error occurred during comparison: {e}", is_match=False), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
