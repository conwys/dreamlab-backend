from hunyuan_service import call_hunyuan_shape_generation_api, save_generated_model
from flask import Flask, jsonify, request, send_from_directory
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
import uuid
import os
from datetime import datetime, timezone
import threading
import time
import shutil

load_dotenv()

sessions_dir = os.getenv("SESSIONS_DIR")

allowed_views = ["front", "back", "left", "right"]

app = Flask(__name__)

# GET - Endpoint to generate a unique session ID
# Creates a directory for the session at ./sessions/<session_id>
@app.route("/api/generate_session_id", methods=["GET"])
def generate_session_id():
    session_id = str(uuid.uuid4())
    session_path = f"{sessions_dir}/{session_id}"
    os.makedirs(session_path, exist_ok=True)
    os.makedirs(f"{session_path}/uploads", exist_ok=True)
    os.makedirs(f"{session_path}/models", exist_ok=True)
    # Write info.txt with session info and timestamp
    info_path = os.path.join(session_path, "info.txt")
    with open(info_path, "w") as f:
        f.write(f"session_id: {session_id}\n")
        f.write(f"created_at: {datetime.now(timezone.utc).isoformat()}\n")
    return jsonify({
        "session_id": session_id
    }), 200

# POST - Endpoint to save image & call the hunyuan_service (which processes image and saves model) # So call your func??
# Uploads will be found at: ./sessions/<session_id>/uploads
@app.route("/api/process_furniture_image/<string:session_id>", methods=["POST"])
def process_image(session_id):

    if not any(f"{view}_image" in request.files and request.files[f"{view}_image"].filename != '' for view in allowed_views):
        return jsonify({"error": "No image files provided for any view"}), 400

    session_upload_dir = os.path.join(sessions_dir, session_id, "uploads")
    
    if not os.path.exists(session_upload_dir):
        return jsonify({"error": "Session_id not found"}), 404
    
    image_paths = save_and_get_image_paths(session_id, request.files)

    if not image_paths:
        return jsonify({"error": "No valid image files were uploaded"}), 400

    caption = request.form.get("caption")

    if caption:
        print(f"INFO: Received user caption: '{caption}'")
    else:
        print("INFO: No user caption provided")

    try:
        model_binary_data, model_filename = call_hunyuan_shape_generation_api(
            image_filepaths=image_paths,
            caption=caption
        )

        file_path, model_public_url = save_generated_model(session_id, model_binary_data, model_filename)

        for view, path in image_paths.items():
            try:
                os.remove(path)
                print(f"INFO: Cleaned up temporary {view} image file: {path}")
            except OSError as e:
                print(f"ERROR: Failed to remove temporary {view} image file {path}: {e}")

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
    session_model_dir = os.path.join(sessions_dir, session_id, "models")
    
    if not os.path.exists(session_model_dir):
        return jsonify({"error": "Session_id not found"}), 404

    models = os.listdir(session_model_dir)
    return jsonify({
        "session_id": session_id,
        "models": models
    }), 200

# This exposes all files under /sessions. 
# So to access models, frontend simply GETs from url <base_url>/sessions/<session_id>/models/<model_filename>
@app.route("/sessions/<path:filename>")
def serve_sessions(filename):
    return send_from_directory("sessions", filename)

def save_and_get_image_paths(session_id, request_files):
    image_paths = {}
    session_upload_dir = os.path.join(sessions_dir, session_id, "uploads")

    for view in allowed_views:
        file_key = f"{view}_image"

        if file_key in request_files and request_files[file_key].filename != "":
            image = request_files[file_key]
            safe_filename = secure_filename(image.filename)
            image_file_path = os.path.join(session_upload_dir, safe_filename)
            image.save(image_file_path)
            image_paths[view] = image_file_path
            print(f"INFO: Saved {view} image to {image_file_path}")
        else:
            print(f"INFO: No {view} image provided for session {session_id}")

    return image_paths

# Removes expired sessions every 5 minutes.
# Sessions expire 1 hour after creation.
def cleanup_expired_sessions():
    while True:
        now = datetime.now(timezone.utc)
        if sessions_dir and os.path.exists(sessions_dir):
            for session_id in os.listdir(sessions_dir):
                session_path = os.path.join(sessions_dir, session_id)
                info_path = os.path.join(session_path, "info.txt")
                try:
                    if os.path.isdir(session_path) and os.path.exists(info_path):
                        with open(info_path, "r") as f:
                            for line in f:
                                if line.startswith("created_at:"):
                                    created_at_str = line.split("created_at:")[1].strip()
                                    created_at = datetime.fromisoformat(created_at_str)
                                    age = (now - created_at).total_seconds()
                                    if age > int(os.getenv('SESSION_EXPIRE_REMOVE_TIME')):
                                        # Remove the session directory                                        
                                        shutil.rmtree(session_path)
                                        print(f"INFO: Deleted expired session: {session_id}")
                                    break
                except Exception as e:
                    print(f"ERROR: Failed to check/delete session {session_id}: {e}")
        # Sleep for 5 minutes
        time.sleep(int(os.getenv('SESSION_EXPIRE_SLEEP_TIME')) or 300)

def start_cleanup_thread():
    t = threading.Thread(target=cleanup_expired_sessions, daemon=True)
    t.start()

if __name__ == "__main__":
    start_cleanup_thread()
    app.run(debug=True)
