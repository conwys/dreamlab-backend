import os
from typing import Dict, Tuple

from dotenv import load_dotenv
from gradio_client import Client, handle_file


# Configuration
load_dotenv()

HUNYUAN_SPACE_ID = os.getenv("HUNYUAN_SPACE_ID")
HUNYUAN_API_NAME = os.getenv("HUNYUAN_API_NAME")
SESSIONS_DIR = os.getenv("SESSIONS_DIR", "sessions")
BASE_URL = os.getenv("APP_BASE_URL", "http://localhost:5000")


# Global client instance (lazy initialisation)
_hunyuan_client: Client | None = None


# Client management
def get_hunyuan_client() -> Client:
    """
    Initialises and returns a Gradio Client instance

    Raises:
        RuntimeError: If the Gradio Client fails to initialise

    Returns:
        Client: The initialised Gradio Client instance
    """
    global _hunyuan_client

    if _hunyuan_client is None:
        if not HUNYUAN_SPACE_ID:
            raise RuntimeError("HUNYUAN_SPACE_ID environment variable is not set")
        try:
            print(f"INFO: Initialising Gradio Client for space ID: {HUNYUAN_SPACE_ID}")
            _hunyuan_client = Client(HUNYUAN_SPACE_ID)
        except Exception as e:
            _hunyuan_client = None
            print(f"ERROR: Failed to initialise Gradio Client for space ID {HUNYUAN_SPACE_ID}: {e}")
            raise RuntimeError(f"Failed to initialise Gradio Client: {e}") from e
    
    return _hunyuan_client


# Model handling
def save_generated_model(session_id: str, model_data: bytes, filename: str) -> Tuple[str, str]:
    """
    Saves the generated model binary data to the specified session directory

    Args:
        session_id (str): The unique identifier for the user session
        model_data (bytes): The binary data of the generated model
        filename (str): The desired filename for the saved model (e.g., 'model.glb')

    Raises:
        IOError: If there's an issue writing the model file

    Returns:
        Tuple[str, str]: A tuple containing the full file path of the saved model and its public URL
    """
    session_models_dir = os.path.join(SESSIONS_DIR, session_id, "models")

    os.makedirs(session_models_dir, exist_ok=True)
    file_path = os.path.join(session_models_dir, filename)

    try:
        with open(file_path, "wb") as f:
            f.write(model_data)
        print(f"INFO: Saved model for session {session_id} to {file_path}")
    except IOError as e:
        print(f"ERROR: Failed to write model data to {file_path}: {e}")
        raise IOError(f"Could not save model file: {e}") from e

    model_url = f"{BASE_URL}/sessions/{session_id}/models/{filename}"
    return file_path, model_url


# Hunyuan API interaction
def _process_image_filepaths(image_filepaths: Dict[str, str], predict_args: Dict) -> Dict:
    """
    Prepares image file paths for the Gradio client's prediction

    Args:
        image_filepaths (Dict[str, str]): A dictionary mapping view names to their corresponding local file paths
        predict_args (Dict): The dictionary of arguments to be passed to client.predict

    Raises:
        ValueError: If no image file paths are provided or if the 'front' image is missing

    Returns:
        Dict: The updated predict_args dictionary with `handle_file` wrappers
    """
    if not image_filepaths:
        raise ValueError("No image file paths provided for Hunyuan API call")

    front_image_path = image_filepaths.get("front")
    if not front_image_path:
        raise ValueError("Front image is required but not provided in image_filepaths")

    # Always use 'image' argument for the front view
    predict_args["image"] = handle_file(front_image_path)

    if len(image_filepaths) == 1:
        print(f"INFO: Passing single image '{os.path.basename(front_image_path)}' to 'image' argument for Hunyuan API")
        return predict_args

    print(f"INFO: Passing {len(image_filepaths)} images to multi-view arguments")

    for view in ["front", "back", "left", "right"]:
        if view in image_filepaths and image_filepaths[view]:
            predict_args[f"mv_image_{view}"] = handle_file(image_filepaths[view])

    return predict_args


def call_hunyuan_shape_generation_api(image_filepaths: Dict[str, str], caption: str | None) -> Tuple[bytes, str]:
    """
    Calls the Hunyuan shape generation API to process images and generate a 3D model

    Args:
        image_filepaths (Dict[str, str]): A dictionary of image file paths
        caption (str | None): An optional text prompt for the model generation

    Raises:
        RuntimeError: If the Hunyuan API call fails or returns an invalid model
        ValueError: If required arguments are missing or invalid

    Returns:
        Tuple[bytes, str]: A tuple containing the binary data of the generated model and its recommended filename
    """
    if not HUNYUAN_API_NAME:
        raise RuntimeError("HUNYUAN_API_NAME environment variable is not set")

    try:
        client = get_hunyuan_client()

        predict_args: Dict = {
            "caption": caption,
            "image": None,
            "mv_image_front": None,
            "mv_image_back": None,
            "mv_image_left": None,
            "mv_image_right": None,
            "api_name": HUNYUAN_API_NAME
        }

        predict_args = _process_image_filepaths(image_filepaths, predict_args)
        
        print(f"INFO: Calling Hunyuan API '{HUNYUAN_API_NAME}' with images: {list(image_filepaths.keys())}")
        result = client.predict(**predict_args)

        generated_model_filepath = result[0]["value"] 
        
        if not generated_model_filepath or not os.path.exists(generated_model_filepath):
            raise RuntimeError(f"Hunyuan API call failed or returned invalid model path: {generated_model_filepath}")
        
        with open(generated_model_filepath, "rb") as f:
            model_binary_data = f.read()

        model_filename = os.path.basename(generated_model_filepath)
        print(f"INFO: Successfully received model '{model_filename}' from Hunyuan API")
        return model_binary_data, model_filename

    except ValueError as ve:
        print(f"ERROR: Input validation failed for Hunyuan API call: {ve}")
        raise ve
    except Exception as e:
        print(f"ERROR: Failed to call Hunyuan API for shape generation: {e}")
        raise RuntimeError(f"Failed to generate shape via Hunyuan API: {e}") from e
