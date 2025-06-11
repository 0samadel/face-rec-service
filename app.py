# face-rec-service/app.py
from flask import Flask, request, jsonify
import face_recognition
import numpy as np
import os

app = Flask(__name__)
UPLOAD_FOLDER = 'temp_uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/generate-embedding', methods=['POST'])
def generate_embedding():
    if 'face' not in request.files:
        return jsonify({"error": "No face image provided"}), 400

    file = request.files['face']
    temp_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(temp_path)

    try:
        image = face_recognition.load_image_file(temp_path)
        # The face_encodings function returns a list of 128-d face encodings
        # We assume only one face is in the enrollment photo
        encodings = face_recognition.face_encodings(image)

        if len(encodings) > 0:
            # Convert numpy array to a plain list for JSON serialization
            embedding = encodings[0].tolist()
            return jsonify({"embedding": embedding}), 200
        else:
            return jsonify({"error": "No face found in the image"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        # Clean up the temporary file
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.route('/compare-faces', methods=['POST'])
def compare_faces():
    if 'face' not in request.files:
        return jsonify({"error": "No face image for comparison provided"}), 400
    
    # The stored embedding is sent as a JSON string in the form data
    stored_embedding_json = request.form.get('stored_embedding')
    if not stored_embedding_json:
        return jsonify({"error": "No stored embedding provided"}), 400

    try:
        # Convert the JSON string back to a list, then to a numpy array
        stored_embedding = np.array(json.loads(stored_embedding_json))
        
        file = request.files['face']
        temp_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(temp_path)

        unknown_image = face_recognition.load_image_file(temp_path)
        unknown_encodings = face_recognition.face_encodings(unknown_image)

        if len(unknown_encodings) > 0:
            unknown_encoding = unknown_encodings[0]
            # compare_faces returns a list of True/False values
            # The tolerance level determines how strict the comparison is. Lower is stricter.
            results = face_recognition.compare_faces([stored_embedding], unknown_encoding, tolerance=0.5)
            is_match = bool(results[0])
            return jsonify({"is_match": is_match}), 200
        else:
            return jsonify({"error": "No face found in the new image"}), 400

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

if __name__ == '__main__':
    # For development, run directly. For production, use Gunicorn.
    # gunicorn --bind 0.0.0.0:5001 app:app
    app.run(host='0.0.0.0', port=5001, debug=True)