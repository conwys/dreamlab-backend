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
def mock_env_vars(monkeypatch):
    """Fixture to set environment variables and module constants"""
    monkeypatch.setenv("HUNYUAN_SPACE_ID", "mock_space_id")
    monkeypatch.setenv("HUNYUAN_API_NAME", "mock_api_name")
    monkeypatch.setenv("SESSIONS_DIR", "test_sessions")
    monkeypatch.setenv("APP_BASE_URL", "http://test-base.com")
    
    monkeypatch.setattr("hunyuan_service.HUNYUAN_SPACE_ID", "mock_space_id")
    monkeypatch.setattr("hunyuan_service.HUNYUAN_API_NAME", "mock_api_name")
    monkeypatch.setattr("hunyuan_service.SESSIONS_DIR", "test_sessions")
    monkeypatch.setattr("hunyuan_service.BASE_URL", "http://test-base.com")
    
    monkeypatch.setattr("hunyuan_service._hunyuan_client", None)


# Tests for get_hunyuan_client
@patch("hunyuan_service.Client")
def test_get_hunyuan_client_initialisation(MockClient):
    """Test the client initialises"""
    mock_instance = MockClient.return_value
    
    client = get_hunyuan_client()
    
    assert client is not None
    MockClient.assert_called_once_with("mock_space_id")


@patch("hunyuan_service.Client")
def test_get_hunyuan_client_singleton(MockClient):
    """Test that subsequent calls return the same client instance"""
    mock_instance = MockClient.return_value
    
    client1 = get_hunyuan_client()
    client2 = get_hunyuan_client()
    
    assert client1 is client2
    MockClient.assert_called_once_with("mock_space_id")


@patch("hunyuan_service.Client", side_effect=Exception("Connection error"))
def test_get_hunyuan_client_initialisation_failure(MockClient):
    """Test error handling when Gradio Client initialisation fails"""
    with pytest.raises(RuntimeError, match="Failed to initialise Gradio Client"):
        get_hunyuan_client()


def test_get_hunyuan_client_no_space_id(monkeypatch):
    """Test error when HUNYUAN_SPACE_ID is not set"""
    monkeypatch.setattr("hunyuan_service.HUNYUAN_SPACE_ID", None)
    monkeypatch.setattr("hunyuan_service._hunyuan_client", None)
    with pytest.raises(RuntimeError, match="HUNYUAN_SPACE_ID environment variable is not set"):
        get_hunyuan_client()


# Tests for save_generated_model
@patch("os.makedirs")
@patch("builtins.open", new_callable=mock_open)
def test_save_generated_model_success(mock_file_open, mock_makedirs):
    """Test successful saving of a model"""
    session_id = "test_session_123"
    model_data = b"binary_model_content"
    filename = "test_model.glb"

    file_path, model_url = save_generated_model(session_id, model_data, filename)

    expected_dir = os.path.join("test_sessions", session_id, "models")
    expected_file_path = os.path.join(expected_dir, filename)
    expected_url = f"http://test-base.com/sessions/{session_id}/models/{filename}"

    mock_makedirs.assert_called_once_with(expected_dir, exist_ok=True)
    mock_file_open.assert_called_once_with(expected_file_path, "wb")
    mock_file_open().write.assert_called_once_with(model_data)

    assert file_path == expected_file_path
    assert model_url == expected_url


@patch("os.makedirs")
@patch("builtins.open", side_effect=IOError("File write failed"))
def test_save_generated_model_file_write_error(mock_file_open, mock_makedirs):
    """Test error handling during file writing"""
    with pytest.raises(IOError, match="Could not save model file"):
        save_generated_model("test_session", b"data", "file.glb")


# Tests for _process_image_filepaths
@patch("hunyuan_service.handle_file")
def test_process_image_filepaths_single_image(mock_handle_file):
    """Test processing with a single image"""
    image_filepaths = {"front": "/path/to/front.png"}
    predict_args = {"caption": "test"}

    mock_handle_file.return_value = MagicMock()

    result_args = _process_image_filepaths(image_filepaths, predict_args)

    assert "image" in result_args
    assert result_args["image"] == mock_handle_file.return_value
    assert "mv_image_front" not in result_args
    mock_handle_file.assert_called_once_with("/path/to/front.png")


@patch("hunyuan_service.handle_file")
def test_process_image_filepaths_multiple_images(mock_handle_file):
    """Test processing with multiple images (front + multi-views)."""
    image_filepaths = {
        "front": "/path/to/front.png",
        "back": "/path/to/back.png",
        "left": "/path/to/left.png",
    }
    predict_args = {"caption": "test"}

    mock_returns = [
        MagicMock(name="handle_file_call_1_image"),
        MagicMock(name="handle_file_call_2_mv_front"),
        MagicMock(name="handle_file_call_3_mv_back"),
        MagicMock(name="handle_file_call_4_mv_left"),
    ]
    mock_handle_file.side_effect = mock_returns

    result_args = _process_image_filepaths(image_filepaths, predict_args)

    assert "image" in result_args
    assert "mv_image_front" in result_args
    assert "mv_image_back" in result_args
    assert "mv_image_left" in result_args
    assert "mv_image_right" not in result_args

    mock_handle_file.assert_any_call("/path/to/front.png")
    mock_handle_file.assert_any_call("/path/to/back.png")
    mock_handle_file.assert_any_call("/path/to/left.png")
    assert mock_handle_file.call_count == 4

    assert result_args["image"] == mock_returns[0]
    assert result_args["mv_image_front"] == mock_returns[1]
    assert result_args["mv_image_back"] == mock_returns[2]
    assert result_args["mv_image_left"] == mock_returns[3]


def test_process_image_filepaths_no_images():
    """Test error when no image file paths are provided"""
    with pytest.raises(ValueError, match="No image file paths provided"):
        _process_image_filepaths({}, {})


def test_process_image_filepaths_no_front_image():
    """Test error when front image is missing"""
    with pytest.raises(ValueError, match="Front image is required"):
        _process_image_filepaths({"back": "path"}, {})


# Tests for call_hunyuan_shape_generation_api
@patch("hunyuan_service.handle_file")
@patch("hunyuan_service.Client")
@patch("hunyuan_service.os.path.exists", return_value=True)
@patch("builtins.open", new_callable=mock_open, read_data=b"mock model content")
def test_call_hunyuan_api_success(mock_file_open, mock_path_exists, MockClient, mock_handle_file):
    """Test successful API call and model return"""
    mock_client_instance = MockClient.return_value
    mock_client_instance.predict.return_value = [{"value": "/tmp/mock_generated_model.glb"}]
    
    mock_handle_file.return_value = MagicMock(name="file_handle")

    image_filepaths = {"front": "/path/to/front.png"}
    caption = "a test caption"

    model_binary_data, model_filename = call_hunyuan_shape_generation_api(image_filepaths, caption)

    assert model_binary_data == b"mock model content"
    assert model_filename == "mock_generated_model.glb"

    mock_client_instance.predict.assert_called_once()
    mock_handle_file.assert_called_once_with("/path/to/front.png")
    mock_file_open.assert_called_once_with("/tmp/mock_generated_model.glb", "rb")


@patch("hunyuan_service.handle_file")
@patch("hunyuan_service.Client")
@patch("hunyuan_service.os.path.exists", return_value=False)
def test_call_hunyuan_api_no_model_file(mock_path_exists, MockClient, mock_handle_file):
    """Test error when API returns a path but file doesn't exist"""
    mock_client_instance = MockClient.return_value
    mock_client_instance.predict.return_value = [{"value": "/tmp/non_existent_model.glb"}]
    
    mock_handle_file.return_value = MagicMock(name="file_handle")

    with pytest.raises(RuntimeError, match="Hunyuan API call failed or returned invalid model path"):
        call_hunyuan_shape_generation_api({"front": "/path/to/front.png"}, "caption")


@patch("hunyuan_service.handle_file")
@patch("hunyuan_service.Client", side_effect=Exception("API call failed"))
def test_call_hunyuan_api_general_error(MockClient, mock_handle_file):
    """Test general exception during API call"""
    with pytest.raises(RuntimeError, match="Failed to generate shape via Hunyuan API"):
        call_hunyuan_shape_generation_api({"front": "/path/to/front.png"}, "caption")


def test_call_hunyuan_api_no_api_name(monkeypatch):
    """Test error when HUNYUAN_API_NAME is not set"""
    monkeypatch.setattr("hunyuan_service.HUNYUAN_API_NAME", None)
    with pytest.raises(RuntimeError, match="HUNYUAN_API_NAME environment variable is not set"):
        call_hunyuan_shape_generation_api({"front": "/path/to/front.png"}, "caption")
