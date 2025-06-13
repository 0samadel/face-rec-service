# face-rec-service/app.py (FINAL - Corrected Distance Calculation)

from flask import Flask, request, jsonify
from deepface import DeepFace
# ✅ IMPORT THE DISTANCE MODULE
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
    _ = DeepFace.represent(np.zeros((100, 100, 3)), model_name=MODEL_NAME, detector_backend=DETECTOR_BACKEND)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model on startup: {e}")

# ... (b64_to_numpy and generate-embedding functions are correct and unchanged) ...
def b64_to_numpy(b64_string):
    if "data:image" in b64_string:
        b64_string = b64_string.split(',')[1]
    img_bytes = base64.b64decode(b64_string)
    pil_img = Image.open(BytesIO(img_bytes)).convert("RGB")
    return np.array(pil_img)

@app.route("/generate-embedding", methods=["POST"])
def generate_embedding():
    try:
        # ... (this function is correct)
    except Exception as e:
        # ...
        
# =========================================================================
# ✅ CORRECTED /compare-faces ROUTE
# =========================================================================
@app.route("/compare-faces", methods=["POST"])
def compare_faces():
    try:
        if "face" not in request.files:
            return jsonify(error="No face image provided for comparison."), 400

        stored_embedding_json = request.form.get("stored_embedding")
        if not stored_embedding_json:
            return jsonify(error="No stored embedding provided."), 400
        
        # 1. Get the embedding of the new face image
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
        
        # 2. Prepare the stored embedding
        stored_embedding = np.array(json.loads(stored_embedding_json))
        
        # 3. ✅ CORRECTED: Calculate the distance using the imported 'dst' module
        distance = dst.findCosineDistance(unknown_embedding, stored_embedding)
        
        # 4. Compare distance to the model's threshold
        threshold = dst.getThreshold(MODEL_NAME, "cosine") # Use the official threshold
        is_match = distance <= threshold
        
        print(f"Comparison Result: Distance={distance:.4f}, Threshold={threshold}, Match={is_match}")
        
        return jsonify(is_match=is_match), 200

    except ValueError as e:
        return jsonify(error=f"Could not process image: {e}", is_match=False), 400
    except Exception as e:
        traceback.print_exc()
        return jsonify(error=f"An internal error occurred during comparison: {e}", is_match=False), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
