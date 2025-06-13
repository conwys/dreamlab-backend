import pytest
import os
from unittest.mock import patch, mock_open, MagicMock

from hunyuan_service import (
    get_hunyuan_client,
    save_generated_model,
    _process_image_filepaths,
    call_hunyuan_shape_generation_api
)


# Fixtures for setup
@pytest.fixture(autouse=True)
def cleanup_hunyuan_client_singleton(monkeypatch):
    """
    Ensure the _hunyuan_client singleton is reset before each test to prevent test interference
    """
    monkeypatch.setattr("hunyuan_service._hunyuan_client", None)


# Tests for get_hunyuan_client
@patch("hunyuan_service.Client")
def test_get_hunyuan_client_initialisation(MockClient):
    """Test the client initialises"""
    mock_space_id = "test_space_id"
    mock_instance = MockClient.return_value

    client = get_hunyuan_client(mock_space_id)

    assert client is not None
    MockClient.assert_called_once_with(mock_space_id)


@patch("hunyuan_service.Client")
def test_get_hunyuan_client_singleton(MockClient):
    """Test that subsequent calls return the same client instance"""
    mock_space_id = "test_space_id"
    mock_instance = MockClient.return_value

    client1 = get_hunyuan_client(mock_space_id)
    client2 = get_hunyuan_client(mock_space_id)

    assert client1 is client2
    MockClient.assert_called_once_with(mock_space_id)


@patch("hunyuan_service.Client", side_effect=Exception("Connection error"))
def test_get_hunyuan_client_initialisation_failure(MockClient):
    """Test error handling when Gradio Client initialisation fails"""
    mock_space_id = "test_space_id"
    with pytest.raises(RuntimeError, match="Failed to initialise Gradio Client"):
        get_hunyuan_client(mock_space_id)


def test_get_hunyuan_client_no_space_id():
    """Test error when HUNYUAN_SPACE_ID is not provided to the function"""
    with pytest.raises(RuntimeError, match="HUNYUAN_SPACE_ID is not provided"):
        get_hunyuan_client("")


# Tests for save_generated_model
@patch("os.makedirs")
@patch("builtins.open", new_callable=mock_open)
def test_save_generated_model_success(mock_file_open, mock_makedirs):
    """Test successful saving of a model"""
    session_id = "test_session_123"
    model_data = b"binary_model_content"
    filename = "test_model.glb"
    sessions_dir = "test_sessions"
    base_url = "http://test-base.com"

    file_path, model_url = save_generated_model(session_id, model_data, filename, sessions_dir, base_url)

    expected_dir = os.path.join(sessions_dir, session_id, "models")
    expected_file_path = os.path.join(expected_dir, filename)
    expected_url = f"{base_url}/sessions/{session_id}/models/{filename}"

    mock_makedirs.assert_called_once_with(expected_dir, exist_ok=True)
    mock_file_open.assert_called_once_with(expected_file_path, "wb")
    mock_file_open().write.assert_called_once_with(model_data)

    assert file_path == expected_file_path
    assert model_url == expected_url


@patch("os.makedirs")
@patch("builtins.open", side_effect=IOError("File write failed"))
def test_save_generated_model_file_write_error(mock_open_fail, mock_makedirs):
    """Test error handling during file writing"""
    with pytest.raises(IOError, match="Could not save model file"):
        save_generated_model("test_session", b"data", "file.glb", "test_sessions", "http://test-base.com")
    mock_makedirs.assert_called_once()


# Tests for _process_image_filepaths
@patch("hunyuan_service.handle_file")
def test_process_image_filepaths_single_image(mock_handle_file):
    """Test processing with a single image (front view only)"""
    image_filepaths = {"front": "/path/to/front.png"}
    predict_args = {"caption": "test", "image": None}
    allowed_views = ["front", "back"]

    mock_handle_file.return_value = MagicMock(name="handle_file_front")

    result_args = _process_image_filepaths(image_filepaths, predict_args, allowed_views)

    assert "image" in result_args
    assert result_args["image"] == mock_handle_file.return_value
    assert result_args.get("mv_image_front") is None
    mock_handle_file.assert_called_once_with("/path/to/front.png")
    assert mock_handle_file.call_count == 1


@patch("hunyuan_service.handle_file")
def test_process_image_filepaths_multiple_images(mock_handle_file):
    """Test processing with multiple images (front + multi-views)"""
    image_filepaths = {
        "front": "/path/to/front.png",
        "back": "/path/to/back.png",
        "left": "/path/to/left.png",
    }
    predict_args = {
        "caption": "test",
        "image": None,
        "mv_image_front": None,
        "mv_image_back": None,
        "mv_image_left": None,
        "mv_image_right": None,
    }
    allowed_views = ["front", "back", "left", "right"]

    mock_main_image = MagicMock(name="handle_file_main_image")
    mock_mv_front = MagicMock(name="handle_file_mv_front")
    mock_mv_back = MagicMock(name="handle_file_mv_back")
    mock_mv_left = MagicMock(name="handle_file_mv_left")

    mock_handle_file.side_effect = [
        mock_main_image,
        mock_mv_front,
        mock_mv_back,
        mock_mv_left,
    ]

    result_args = _process_image_filepaths(image_filepaths, predict_args, allowed_views)

    assert "image" in result_args
    assert result_args["image"] is mock_main_image

    assert "mv_image_front" in result_args
    assert result_args["mv_image_front"] is mock_mv_front

    assert "mv_image_back" in result_args
    assert result_args["mv_image_back"] is mock_mv_back
    
    assert "mv_image_left" in result_args
    assert result_args["mv_image_left"] is mock_mv_left
    
    assert result_args.get("mv_image_right") is None

    assert mock_handle_file.call_count == 4
    mock_handle_file.assert_any_call("/path/to/front.png")
    mock_handle_file.assert_any_call("/path/to/front.png")
    mock_handle_file.assert_any_call("/path/to/back.png")
    mock_handle_file.assert_any_call("/path/to/left.png")


def test_process_image_filepaths_no_images():
    """Test error when no image file paths are provided"""
    with pytest.raises(ValueError, match="No image file paths provided for Hunyuan API call"):
        _process_image_filepaths({}, {}, [])


def test_process_image_filepaths_no_front_image():
    """Test error when front image is missing."""
    with pytest.raises(ValueError, match="Front image is required but not provided in image_filepaths"):
        _process_image_filepaths({"back": "/path/to/back.png"}, {}, ["front", "back"])


# Tests for call_hunyuan_shape_generation_api
@patch("hunyuan_service.get_hunyuan_client")
@patch("hunyuan_service.handle_file")
@patch("hunyuan_service.os.path.exists", return_value=True)
@patch("builtins.open", new_callable=mock_open, read_data=b"mock model content")
def test_call_hunyuan_api_success(mock_file_open, mock_os_path_exists, mock_handle_file, mock_get_client):
    """Test successful API call and model return"""
    mock_client_instance = MagicMock()
    mock_get_client.return_value = mock_client_instance
    mock_client_instance.predict.return_value = [{"value": "/tmp/mock_generated_model.glb"}]

    mock_handle_file.return_value = MagicMock(name="file_handle")

    image_filepaths = {"front": "/path/to/front.png"}
    caption = "a test caption"
    hunyuan_space_id = "test_space"
    hunyuan_api_name = "test_api"
    allowed_views = ["front"]

    model_binary_data, model_filename = call_hunyuan_shape_generation_api(
        image_filepaths, caption, hunyuan_space_id, hunyuan_api_name, allowed_views
    )

    assert model_binary_data == b"mock model content"
    assert model_filename == "mock_generated_model.glb"

    mock_get_client.assert_called_once_with(hunyuan_space_id)
    mock_client_instance.predict.assert_called_once_with(
        caption=caption,
        image=mock_handle_file.return_value,
        mv_image_front=None,
        mv_image_back=None,
        mv_image_left=None,
        mv_image_right=None,
        api_name=hunyuan_api_name
    )
    mock_handle_file.assert_called_once_with("/path/to/front.png")
    mock_file_open.assert_called_once_with("/tmp/mock_generated_model.glb", "rb")


@patch("hunyuan_service.get_hunyuan_client")
@patch("hunyuan_service.handle_file")
@patch("hunyuan_service.os.path.exists", return_value=False)
def test_call_hunyuan_api_no_model_file(mock_os_path_exists, mock_handle_file, mock_get_client):
    """Test error when API returns a path but file doesn't exist"""
    mock_client_instance = MagicMock()
    mock_get_client.return_value = mock_client_instance
    mock_client_instance.predict.return_value = [{"value": "/tmp/non_existent_model.glb"}]

    mock_handle_file.return_value = MagicMock(name="file_handle")

    with pytest.raises(RuntimeError, match="Hunyuan API call failed or returned invalid model path"):
        call_hunyuan_shape_generation_api(
            {"front": "/path/to/front.png"}, "caption", "test_space", "test_api", ["front"]
        )
    mock_get_client.assert_called_once_with("test_space")
    mock_client_instance.predict.assert_called_once()
    mock_os_path_exists.assert_called_once_with("/tmp/non_existent_model.glb")


@patch("hunyuan_service.get_hunyuan_client", side_effect=Exception("API call failed"))
def test_call_hunyuan_api_general_error(mock_get_client):
    """Test general exception during API call"""
    with pytest.raises(RuntimeError, match="Failed to generate shape via Hunyuan API"):
        call_hunyuan_shape_generation_api(
            {"front": "/path/to/front.png"}, "caption", "test_space", "test_api", ["front"]
        )
    mock_get_client.assert_called_once_with("test_space")


def test_call_hunyuan_api_no_api_name():
    """Test error when HUNYUAN_API_NAME is not provided to the function"""
    with pytest.raises(RuntimeError, match="HUNYUAN_API_NAME is not provided"):
        call_hunyuan_shape_generation_api(
            {"front": "/path/to/front.png"}, "caption", "test_space", "", ["front"]
        )
