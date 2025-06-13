import pytest
import os
import json
import shutil
from unittest.mock import patch, MagicMock
from flask import Response
from io import BytesIO
from werkzeug.datastructures import FileStorage

from app import create_app
from config import TestingConfig


# Fixtures for setup
@pytest.fixture
def app():
    """Create and configure a new app instance for each test"""
    class CustomTestingConfig(TestingConfig):
        SESSIONS_DIR = "test_sessions_tmp"
        HUNYUAN_SPACE_ID = "test_space"
        HUNYUAN_API_NAME = "test_api"
        APP_BASE_URL = "http://test-base.com"
        SESSION_EXPIRE_REMOVE_SECONDS = 1
        SESSION_CLEANUP_INTERVAL_SECONDS = 0.1
        ALLOWED_VIEWS = ["front", "back", "left", "right"]

    app = create_app(config_class=CustomTestingConfig)

    if os.path.exists(app.config["SESSIONS_DIR"]):
        shutil.rmtree(app.config["SESSIONS_DIR"])
    os.makedirs(app.config["SESSIONS_DIR"])

    yield app

    if os.path.exists(app.config["SESSIONS_DIR"]):
        shutil.rmtree(app.config["SESSIONS_DIR"])


@pytest.fixture
def client(app):
    """A test client for the app"""
    return app.test_client()


@pytest.fixture(autouse=True)
def stop_cleanup_thread(monkeypatch):
    """Prevent the background cleanup thread from starting during tests"""
    monkeypatch.setattr("threading.Thread", MagicMock())
    monkeypatch.setattr("utils.session_helpers.cleanup_expired_sessions", MagicMock())


@pytest.fixture
def mock_image_file():
    """Create a mock image file"""
    return FileStorage(
        stream=BytesIO(b"fake image data"),
        filename="test_image.jpg",
        content_type="image/jpeg"
    )

@pytest.fixture
def mock_side_image_file():
    """Create a mock side image file"""
    return FileStorage(
        stream=BytesIO(b"fake side image data"),
        filename="test_side.jpg",
        content_type="image/jpeg"
    )


# Tests for /api/generate_session_id endpoint
@patch("api.routes._create_session_directories")
@patch("api.routes._write_session_info")
def test_generate_session_id_success(mock_write_session_info, mock_create_session_directories, client):
    """Test successful generation of session ID and directory creation"""
    response = client.get("/api/generate_session_id")
    data = json.loads(response.data)

    assert response.status_code == 200
    assert "session_id" in data
    assert len(data["session_id"]) == 36 # UUID format

    session_path = os.path.join(client.application.config["SESSIONS_DIR"], data["session_id"])
    mock_create_session_directories.assert_called_once_with(session_path)
    mock_write_session_info.assert_called_once_with(session_path, data["session_id"])


@patch("api.routes._create_session_directories", side_effect=OSError("Disk full"))
def test_generate_session_id_dir_creation_failure(mock_create_session_directories_fail, client):
    """Test error handling when session directory creation fails"""
    response = client.get("/api/generate_session_id")
    data = json.loads(response.data)

    assert response.status_code == 500
    assert "error" in data
    assert data["error"] == "Failed to create session"
    mock_create_session_directories_fail.assert_called()


# Tests for /api/process_furniture_image/<session_id> endpoint
@patch("api.routes.save_generated_model")
@patch("api.routes.call_hunyuan_shape_generation_api")
@patch("api.routes._save_and_get_image_paths")
def test_process_image_success_single_image(
    mock_save_paths, mock_call_api, mock_save_model, client, app, mock_image_file
):
    """Test successful image processing with a single front image"""
    session_id = "test_session_single"
    session_dir = os.path.join(app.config["SESSIONS_DIR"], session_id)
    os.makedirs(session_dir, exist_ok=True)
    
    mock_save_paths.return_value = ["/path/to/front.jpg"]
    mock_call_api.return_value = (b"mock_model_data", "generated_model.glb")
    mock_save_model.return_value = (
        f"{session_dir}/models/generated_model.glb",
        f"{app.config['APP_BASE_URL']}/sessions/{session_id}/models/generated_model.glb"
    )

    response = client.post(
        f"/api/process_furniture_image/{session_id}",
        data={"caption": "A single chair", "front_image": mock_image_file},
        content_type="multipart/form-data"
    )
    data = json.loads(response.data)

    assert response.status_code == 200
    assert data["message"] == "Image processed and model generated successfully"
    assert data["filename"] == "generated_model.glb"
    assert data["model_url"].endswith(f"/sessions/{session_id}/models/generated_model.glb")
    mock_save_paths.assert_called_once()
    mock_call_api.assert_called_once_with(
        image_filepaths=["/path/to/front.jpg"],
        caption="A single chair",
        hunyuan_space_id=app.config["HUNYUAN_SPACE_ID"],
        hunyuan_api_name=app.config["HUNYUAN_API_NAME"],
        allowed_views=app.config["ALLOWED_VIEWS"]
    )
    mock_save_model.assert_called_once()


@patch("api.routes.save_generated_model")
@patch("api.routes.call_hunyuan_shape_generation_api")
@patch("api.routes._save_and_get_image_paths")
def test_process_image_success_multiple_images(
    mock_save_paths, mock_call_api, mock_save_model, client, app, mock_image_file, mock_side_image_file
):
    """Test successful image processing with multiple images (front and side)"""
    session_id = "test_session_multi"
    session_dir = os.path.join(app.config["SESSIONS_DIR"], session_id)
    os.makedirs(session_dir, exist_ok=True)

    saved_paths = ["/path/to/front.jpg", "/path/to/side.jpg"]
    mock_save_paths.return_value = saved_paths
    mock_call_api.return_value = (b"mock_model_data_multi", "multi_view_model.glb")
    mock_save_model.return_value = (
        f"{session_dir}/models/multi_view_model.glb",
        f"{app.config['APP_BASE_URL']}/sessions/{session_id}/models/multi_view_model.glb"
    )

    response = client.post(
        f"/api/process_furniture_image/{session_id}",
        data={
            "caption": "A chair with multiple views",
            "front_image": mock_image_file,
            "side_image": mock_side_image_file,
        },
        content_type="multipart/form-data"
    )
    data = json.loads(response.data)

    assert response.status_code == 200
    assert data["message"] == "Image processed and model generated successfully"
    mock_call_api.assert_called_once_with(
        image_filepaths=saved_paths,
        caption="A chair with multiple views",
        hunyuan_space_id=app.config["HUNYUAN_SPACE_ID"],
        hunyuan_api_name=app.config["HUNYUAN_API_NAME"],
        allowed_views=app.config["ALLOWED_VIEWS"]
    )


@patch("api.routes.call_hunyuan_shape_generation_api", side_effect=Exception("API Error"))
@patch("api.routes._save_and_get_image_paths", return_value=["/path/to/front.jpg"])
def test_process_image_api_failure(mock_save_paths, mock_call_api, client, app, mock_image_file):
    """Test failure when the external Hunyuan API call raises an exception"""
    session_id = "test_session_api_fail"
    os.makedirs(os.path.join(app.config["SESSIONS_DIR"], session_id), exist_ok=True)

    response = client.post(
        f"/api/process_furniture_image/{session_id}",
        data={"front_image": mock_image_file},
        content_type="multipart/form-data"
    )
    data = json.loads(response.data)

    assert response.status_code == 500
    assert data["error"] == "An internal error occurred while processing the image"
    mock_call_api.assert_called_once()


def test_process_image_session_not_found(client, mock_image_file):
    """Test error when the provided session_id does not exist"""
    response = client.post(
        "/api/process_furniture_image/non_existent_session",
        data={"front_image": mock_image_file},
        content_type="multipart/form-data"
    )
    data = json.loads(response.data)

    assert response.status_code == 404
    assert data["error"] == "Session ID not found"

def test_process_image_no_front_image(client, app):
    """Test error when the required 'front_image' is missing from the request"""
    session_id = "test_session_no_front"
    os.makedirs(os.path.join(app.config["SESSIONS_DIR"], session_id), exist_ok=True)

    response = client.post(
        f"/api/process_furniture_image/{session_id}",
        data={"caption": "This request will fail"}, # No files
        content_type="multipart/form-data"
    )
    data = json.loads(response.data)

    assert response.status_code == 400
    assert data["error"] == "The 'front_image' is required"


@patch("api.routes._save_and_get_image_paths", return_value=[])
def test_process_image_no_valid_image_uploaded(mock_save_paths, client, app, mock_image_file):
    """Test error when _save_and_get_image_paths returns an empty list"""
    session_id = "test_session_no_valid_img"
    os.makedirs(os.path.join(app.config["SESSIONS_DIR"], session_id), exist_ok=True)

    response = client.post(
        f"/api/process_furniture_image/{session_id}",
        data={"front_image": mock_image_file},
        content_type="multipart/form-data"
    )
    data = json.loads(response.data)

    assert response.status_code == 400
    assert data["error"] == "No valid image files were uploaded"
    mock_save_paths.assert_called_once()


# Tests for /api/session_models/<session_id> endpoint
@patch("os.listdir", return_value=["model1.glb", "model2.gltf"])
def test_get_session_models_success(mock_listdir, client):
    """Test retrieval of session models"""
    session_id = "test_session_models"
    sessions_dir = client.application.config["SESSIONS_DIR"]
    session_path = os.path.join(sessions_dir, session_id)
    session_models_dir = os.path.join(session_path, "models")
    
    os.makedirs(session_models_dir, exist_ok=True)

    response = client.get(f"/api/session_models/{session_id}")
    data = json.loads(response.data)

    assert response.status_code == 200
    assert "models" in data
    assert data["models"] == ["model1.glb", "model2.gltf"]

    mock_listdir.assert_called_once_with(session_models_dir)


@patch("os.path.exists", return_value=False)
def test_get_session_models_session_not_found(mock_exists, client):
    """Test case when session models directory does not exist"""
    response = client.get("/api/session_models/non_existent_session")
    data = json.loads(response.data)

    assert response.status_code == 404
    assert "error" in data
    assert data["error"] == "Session ID not found or models directory missing"
    mock_exists.assert_called_once_with(os.path.join(client.application.config["SESSIONS_DIR"], "non_existent_session", "models"))


# Tests for /sessions/<path:filename> static file serving
@patch("api.routes.send_from_directory")
def test_serve_sessions(mock_send_from_directory, client):
    """Test the static file serving endpoint"""
    filename = "test_session/uploads/image.png"

    mock_response_content = b"dummy_file_content"
    mock_response_mimetype = "image/png"

    flask_response_object = Response(
        mock_response_content,
        mimetype=mock_response_mimetype,
        status=200
    )

    mock_send_from_directory.return_value = flask_response_object

    response = client.get(f"/api/sessions/{filename}")

    assert response.status_code == 200
    assert response.data == mock_response_content
    assert response.mimetype == mock_response_mimetype

    mock_send_from_directory.assert_called_once_with(client.application.config["SESSIONS_DIR"], filename)
