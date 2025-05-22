from hunyuan_service import call_hunyuan_shape_generation_api, save_generated_model
from flask import Flask, jsonify, request, send_from_directory
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
import uuid
import os


load_dotenv()

sessions_dir = os.getenv("SESSIONS_DIR")

app = Flask(__name__)

print("running")

# GET - Endpoint to generate a unique session ID
# Creates a directory for the session at ./sessions/<session_id>
@app.route("/api/generate_session_id", methods=["GET"])
def generate_session_id():
    session_id = str(uuid.uuid4())
    os.makedirs(f"{sessions_dir}/{session_id}", exist_ok=True)
    os.makedirs(f"{sessions_dir}/{session_id}/uploads", exist_ok=True)
    os.makedirs(f"{sessions_dir}/{session_id}/models", exist_ok=True)
    return jsonify({
        "session_id": session_id
    }), 200

# POST - Endpoint to save image & call the hunyuan_service (which processes image and saves model) # So call your func??
# Uploads will be found at: ./sessions/<session_id>/uploads
@app.route("/api/process_furniture_image/<string:session_id>", methods=["POST"])
def process_image(session_id):
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    image = request.files['image']
    session_dir = os.path.join(sessions_dir, session_id, "uploads")
    if not os.path.exists(session_dir):
        return jsonify({"error": "Session_id not found"}), 404

    safe_filename = secure_filename(image.filename)
    image_file_path = os.path.join(session_dir, safe_filename)
    image.save(image_file_path)

    try:
        model_binary_data, model_filename = call_hunyuan_shape_generation_api(
            image_filepath=image_file_path,
            caption=f"3D model generated from {safe_filename}"
        )

        file_path, model_public_url = save_generated_model(session_id, model_binary_data, model_filename)

        try:
            os.remove(image_file_path)
            print(f"INFO: Cleaned up temporary image file: {image_file_path}")
        except OSError as e:
            print(f"ERROR: Failed to remove temporary image file {image_file_path}: {e}")

        return jsonify({
            "message": "Image saved successfully",
            "session_id": session_id,
            "filename": os.path.basename(file_path),
            "model_url": model_public_url
        }), 200

    except Exception as e:
        print(f"ERROR: Failed processing image because {e}")
        return jsonify({"error": "An internal error occurred while processing the image"}), 500


# GET - Endpoint to return a list of file names for models generated. 
# Models will be found at: ./sessions/<session_id>/models
@app.route("/api/session_models/<string:session_id>", methods=["GET"])
def get_session_models(session_id):
    session_dir = os.path.join(sessions_dir, session_id, "models")
    if not os.path.exists(session_dir):
        return jsonify({"error": "Session_id not found"}), 404

    models = os.listdir(session_dir)
    return jsonify({
        "session_id": session_id,
        "models": models
    }), 200


# This exposes all files under /sessions. 
# So to access models, frontend simply GETs from url <base_url>/sessions/<session_id>/models/<model_filename>
@app.route("/sessions/<path:filename>")
def serve_sessions(filename):
    return send_from_directory('sessions', filename)


if __name__ == "__main__":
    app.run(debug=True)
