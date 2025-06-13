import os
import uuid

from flask import jsonify, request, current_app, send_from_directory
from werkzeug.utils import secure_filename

from . import api_bp
from hunyuan_service import call_hunyuan_shape_generation_api, save_generated_model
from utils.session_helpers import _create_session_directories, _write_session_info, _save_and_get_image_paths


@api_bp.route("/generate_session_id", methods=["GET"])
def generate_session_id():
    """
    Generates a unique session ID and creates the necessary directories for it

    Returns:
        JSON: A dictionary containing the generated `session_id`.
    """
    session_id = str(uuid.uuid4())
    session_path = os.path.join(current_app.config["SESSIONS_DIR"], session_id)

    try:
        _create_session_directories(session_path)
        _write_session_info(session_path, session_id)
        return jsonify({"session_id": session_id}), 200
    except Exception as e:
        current_app.logger.error(f"Failed to generate session ID or create directories: {e}")
        return jsonify({"error": "Failed to create session"}), 500


@api_bp.route("/process_furniture_image/<string:session_id>", methods=["POST"])
def process_image(session_id: str):
    """
    Processes uploaded furniture images for a given session, calls the Hunyuan API to generate a 3D model, and saves the model

    Args:
        session_id (str): The unique identifier for the user session

    Returns:
        JSON: A dictionary containing model information (`filename`, `model_url`) and a success message, or an error message
    """
    caption = request.form.get("caption")
    current_app.logger.info(f"Received request for session {session_id}. Caption: '{caption or 'N/A'}'")

    session_base_dir = os.path.join(current_app.config["SESSIONS_DIR"], session_id)
    if not os.path.exists(session_base_dir):
        current_app.logger.warning(f"Session ID not found: {session_id}")
        return jsonify({"error": "Session ID not found"}), 404

    if "front_image" not in request.files or request.files["front_image"].filename == "":
        return jsonify({"error": "The 'front_image' is required"}), 400

    image_paths = _save_and_get_image_paths(session_id, request.files)

    if not image_paths:
        current_app.logger.warning(f"No valid images uploaded for session {session_id}")
        return jsonify({"error": "No valid image files were uploaded"}), 400

    try:
        model_binary_data, model_filename = call_hunyuan_shape_generation_api(
            image_filepaths=image_paths,
            caption=caption,
            hunyuan_space_id=current_app.config["HUNYUAN_SPACE_ID"],
            hunyuan_api_name=current_app.config["HUNYUAN_API_NAME"],
            allowed_views=current_app.config["ALLOWED_VIEWS"]
        )

        file_path, model_public_url = save_generated_model(
            session_id, model_binary_data, model_filename,
            sessions_dir=current_app.config["SESSIONS_DIR"],
            base_url=current_app.config["APP_BASE_URL"]
        )

        return jsonify({
            "message": "Image processed and model generated successfully",
            "session_id": session_id,
            "filename": os.path.basename(file_path),
            "model_url": model_public_url
        }), 200

    except Exception as e:
        current_app.logger.error(f"Failed processing image for session {session_id}: {e}")
        return jsonify({"error": "An internal error occurred while processing the image"}), 500


@api_bp.route("/session_models/<string:session_id>", methods=["GET"])
def get_session_models(session_id: str):
    """
    Retrieves a list of 3D models generated for a specific session

    Args:
        session_id (str): The unique identifier for the user session

    Returns:
        JSON: A dictionary containing the `session_id` and a list of `models` (filenames), or an error message
    """
    sessions_dir = current_app.config["SESSIONS_DIR"]
    session_path = os.path.join(sessions_dir, session_id)
    session_models_dir = os.path.join(session_path, "models")

    if not os.path.exists(session_models_dir):
        current_app.logger.warning(f"Session model directory not found: {session_id}")
        return jsonify({"error": "Session ID not found or models directory missing"}), 404

    try:
        models = os.listdir(session_models_dir)
        return jsonify({
            "session_id": session_id,
            "models": models
        }), 200
    except Exception as e:
        current_app.logger.error(f"Failed to list models for session {session_id}: {e}")
        return jsonify({"error": "An internal error occurred while retrieving models"}), 500


@api_bp.route("/sessions/<path:filename>")
def serve_sessions(filename: str):
    """
    Serves static files (e.g., uploaded images, generated 3D models) from session directories

    Args:
        filename (str): The path to the file relative to the SESSIONS_DIR (e.g., 'session_id/uploads/image.png' or 'session_id/models/model.glb')

    Returns:
        File: The requested file
    """
    sessions_dir = current_app.config["SESSIONS_DIR"]

    # Ensure the filename does not contain '..' for security
    if ".." in filename:
        current_app.logger.warning(f"Attempted path traversal detected: {filename}")
        return "Forbidden", 403

    try:
        return send_from_directory(sessions_dir, filename)
    except Exception as e:
        current_app.logger.error(f"Error serving file {filename} from {sessions_dir}: {e}")
        return "File not found or access denied", 404
