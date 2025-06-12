import os
import shutil
import threading
import time
import uuid
from datetime import datetime, timezone

from dotenv import load_dotenv
from flask import Flask, jsonify, request, send_from_directory
from werkzeug.utils import secure_filename

from hunyuan_service import call_hunyuan_shape_generation_api, save_generated_model


# Configuration
load_dotenv()

SESSIONS_DIR = os.getenv("SESSIONS_DIR")
ALLOWED_VIEWS = ["front", "back", "left", "right"]
SESSION_EXPIRE_REMOVE_SECONDS = int(os.getenv('SESSION_EXPIRE_REMOVE_TIME', 3600))
SESSION_CLEANUP_INTERVAL_SECONDS = int(os.getenv('SESSION_EXPIRE_SLEEP_TIME', 300))

app = Flask(__name__)


# Helper functions
def _create_session_directories(session_path: str) -> None:
    """Creates the necessary directories for a new session"""
    os.makedirs(session_path, exist_ok=True)
    os.makedirs(os.path.join(session_path, "uploads"), exist_ok=True)
    os.makedirs(os.path.join(session_path, "models"), exist_ok=True)
    print(f"INFO: Created session directories for {session_path}")


def _write_session_info(session_path: str, session_id: str) -> None:
    """Writes session metadata to an info.txt file"""
    info_path = os.path.join(session_path, "info.txt")
    with open(info_path, "w") as f:
        f.write(f"session_id: {session_id}\n")
        f.write(f"created_at: {datetime.now(timezone.utc).isoformat()}\n")
    print(f"INFO: Wrote info.txt for session {session_id}")


def _save_and_get_image_paths(session_id: str, request_files: dict) -> dict:
    """Saves uploaded images for a given session and returns a dictionary of saved image paths"""
    image_paths = {}
    session_upload_dir = os.path.join(SESSIONS_DIR, session_id, "uploads")

    if not os.path.exists(session_upload_dir):
        print(f"ERROR: Upload directory not found for session {session_id}: {session_upload_dir}")
        return {}

    for view in ALLOWED_VIEWS:
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


def cleanup_expired_sessions() -> None:
    """Periodically removes session directories that have expired based on their creation timestamp"""
    while True:
        now = datetime.now(timezone.utc)
        if not os.path.exists(SESSIONS_DIR):
            print(f"WARNING: Sessions directory not found: {SESSIONS_DIR}")
            time.sleep(SESSION_CLEANUP_INTERVAL_SECONDS)
            continue

        for session_id in os.listdir(SESSIONS_DIR):
            session_path = os.path.join(SESSIONS_DIR, session_id)
            info_path = os.path.join(session_path, "info.txt")

            if not os.path.isdir(session_path) or not os.path.exists(info_path):
                print(f"INFO: Skipping non-session directory or missing info.txt: {session_path}")
                continue

            try:
                created_at = None
                with open(info_path, "r") as f:
                    for line in f:
                        if line.startswith("created_at:"):
                            created_at_str = line.split("created_at:")[1].strip()
                            created_at = datetime.fromisoformat(created_at_str)
                            break
                
                if created_at:
                    age = (now - created_at).total_seconds()
                    if age > SESSION_EXPIRE_REMOVE_SECONDS:
                        shutil.rmtree(session_path)
                        print(f"INFO: Deleted expired session: {session_id} (age: {age:.0f}s)")
                else:
                    print(f"WARNING: Could not find 'created_at' in info.txt for session: {session_id}. Skipping cleanup.")

            except Exception as e:
                print(f"ERROR: Failed to check/delete session {session_id}: {e}")
        
        time.sleep(SESSION_CLEANUP_INTERVAL_SECONDS)


def start_cleanup_thread() -> None:
    """Starts the session cleanup process in a separate daemon thread"""
    cleanup_thread = threading.Thread(target=cleanup_expired_sessions, daemon=True)
    cleanup_thread.start()
    print("INFO: Started background session cleanup thread.")


# API endpoints
@app.route("/api/generate_session_id", methods=["GET"])
def generate_session_id_endpoint():
    """
    GET /api/generate_session_id
    Generates a unique session ID and creates corresponding directories
    """
    session_id = str(uuid.uuid4())
    session_path = os.path.join(SESSIONS_DIR, session_id)

    try:
        _create_session_directories(session_path)
        _write_session_info(session_path, session_id)
        return jsonify({"session_id": session_id}), 200
    except Exception as e:
        print(f"ERROR: Failed to generate session ID or create directories: {e}")
        return jsonify({"error": "Failed to create session"}), 500


@app.route("/api/process_furniture_image/<string:session_id>", methods=["POST"])
def process_image_endpoint(session_id: str):
    """
    POST /api/process_furniture_image/<session_id>
    Receives furniture images, processes them via hunyuan_service, and saves the generated model
    """
    caption = request.form.get("caption")
    print(f"INFO: Received request for session {session_id}. Caption: '{caption or 'N/A'}'")

    session_base_dir = os.path.join(SESSIONS_DIR, session_id)
    if not os.path.exists(session_base_dir):
        print(f"WARNING: Session ID not found: {session_id}")
        return jsonify({"error": "Session ID not found"}), 404

    if "front_image" not in request.files or request.files["front_image"].filename == "":
        return jsonify({"error": "The 'front_image' is required"}), 400

    image_paths = _save_and_get_image_paths(session_id, request.files)

    if not image_paths:
        return jsonify({"error": "No valid image files were uploaded"}), 400

    try:
        model_binary_data, model_filename = call_hunyuan_shape_generation_api(
            image_filepaths=image_paths,
            caption=caption
        )

        file_path, model_public_url = save_generated_model(
            session_id, model_binary_data, model_filename
        )

        return jsonify({
            "message": "Image processed and model generated successfully",
            "session_id": session_id,
            "filename": os.path.basename(file_path),
            "model_url": model_public_url
        }), 200

    except Exception as e:
        print(f"ERROR: Failed processing image for session {session_id}: {e}")
        return jsonify({"error": "An internal error occurred while processing the image"}), 500


@app.route("/api/session_models/<string:session_id>", methods=["GET"])
def get_session_models_endpoint(session_id: str):
    """
    GET /api/session_models/<session_id>
    Returns a list of generated model filenames for a given session
    """
    session_model_dir = os.path.join(SESSIONS_DIR, session_id, "models")
    
    if not os.path.exists(session_model_dir):
        print(f"WARNING: Session model directory not found: {session_id}")
        return jsonify({"error": "Session ID not found or models directory missing"}), 404

    try:
        models = os.listdir(session_model_dir)
        return jsonify({
            "session_id": session_id,
            "models": models
        }), 200
    except Exception as e:
        print(f"ERROR: Failed to list models for session {session_id}: {e}")
        return jsonify({"error": "An internal error occurred while retrieving models"}), 500


@app.route("/sessions/<path:filename>")
def serve_sessions(filename: str):
    """
    GET /sessions/<path:filename>
    Serves static files from the 'sessions' directory
    """
    return send_from_directory(SESSIONS_DIR, filename)


# Main
if __name__ == "__main__":
    start_cleanup_thread()
    app.run(debug=True) # TODO Change this for production
