import os
import shutil
import time
from datetime import datetime, timezone

from flask import current_app
from werkzeug.utils import secure_filename


def _create_session_directories(session_path: str) -> None:
    """
    Creates the necessary directories for a new session:
    - The main session directory
    - 'uploads' subdirectory for raw image uploads
    - 'models' subdirectory for generated 3D models
    """
    os.makedirs(session_path, exist_ok=True)
    os.makedirs(os.path.join(session_path, "uploads"), exist_ok=True)
    os.makedirs(os.path.join(session_path, "models"), exist_ok=True)
    current_app.logger.info(f"Created session directories for {session_path}")


def _write_session_info(session_path: str, session_id: str) -> None:
    """Writes session metadata (session_id, creation timestamp) to an 'info.txt' file within the session's root directory"""
    info_path = os.path.join(session_path, "info.txt")
    with open(info_path, "w") as f:
        f.write(f"session_id: {session_id}\n")
        f.write(f"created_at: {datetime.now(timezone.utc).isoformat()}\n")
    current_app.logger.info(f"Wrote info.txt for session {session_id}")


def _save_and_get_image_paths(session_id: str, request_files: dict) -> dict:
    """
    Saves uploaded images for a given session and returns a dictionary mapping view names to the absolute paths of the saved images

    Args:
        session_id (str): The ID of the session
        request_files (dict): The `request.files` object from Flask, containing uploaded files

    Returns:
        dict: A dictionary where keys are view names (e.g., 'front') and values are the file paths of the saved images
        Returns an empty dict if no valid images were uploaded or the session directory doesn't exist
    """
    image_paths = {}
    session_upload_dir = os.path.join(current_app.config["SESSIONS_DIR"], session_id, "uploads")

    if not os.path.exists(session_upload_dir):
        current_app.logger.error(f"Upload directory not found for session {session_id}: {session_upload_dir}")
        return {}

    allowed_views = current_app.config.get("ALLOWED_VIEWS", [])
    if not allowed_views:
        current_app.logger.warning("ALLOWED_VIEWS is not configured in app.config. No images will be saved")
        return {}

    found_any_image = False
    for view in allowed_views:
        file_key = f"{view}_image"

        if file_key in request_files and request_files[file_key].filename != "":
            image = request_files[file_key]
            safe_filename = secure_filename(image.filename)
            image_file_path = os.path.join(session_upload_dir, safe_filename)
            image.save(image_file_path)
            image_paths[view] = image_file_path
            current_app.logger.info(f"Saved {view} image to {image_file_path}")
            found_any_image = True
        else:
            current_app.logger.info(f"No {view} image provided for session {session_id}")

    if not found_any_image:
        return {}

    return image_paths


def cleanup_expired_sessions(sessions_dir: str, expire_seconds: int, cleanup_interval: int) -> None:
    """
    Periodically removes session directories that have expired based on their creation timestamp
    This function is intended to run in a separate thread

    Args:
        sessions_dir (str): The base directory where all session folders are stored
        expire_seconds (int): The number of seconds after which a session is considered expired
        cleanup_interval (int): The time in seconds to wait between cleanup scans
    """
    while True:
        now = datetime.now(timezone.utc)
        if not os.path.exists(sessions_dir):
            current_app.logger.warning(f"Sessions directory not found: {sessions_dir}. Waiting for it to appear")
            time.sleep(cleanup_interval)
            continue

        for session_id in os.listdir(sessions_dir):
            session_path = os.path.join(sessions_dir, session_id)
            info_path = os.path.join(session_path, "info.txt")

            if not os.path.isdir(session_path) or not os.path.exists(info_path):
                current_app.logger.debug(f"Skipping non-session directory or missing info.txt: {session_path}")
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
                    if age > expire_seconds:
                        shutil.rmtree(session_path)
                        current_app.logger.info(f"Deleted expired session: {session_id} (age: {age:.0f}s)")
                else:
                    current_app.logger.warning(f"Could not find 'created_at' in info.txt for session: {session_id}. Skipping cleanup")

            except Exception as e:
                current_app.logger.error(f"Failed to check/delete session {session_id}: {e}")

        time.sleep(cleanup_interval)
