import os
import logging
from typing import Dict, Tuple, Optional

from gradio_client import Client, handle_file


# Logger
logger = logging.getLogger(__name__)


# Global client instance (lazy initialisation)
_hunyuan_client: Optional[Client] = None


# Client management
def get_hunyuan_client(hunyuan_space_id: str) -> Client:
    """
    Initialises a Gradio Client instance

    Args:
        hunyuan_space_id (str): Hunyuan space identifier

    Raises:
        RuntimeError: If the Gradio Client fails to initialise

    Returns:
        Client: The initialised Gradio Client instance
    """
    global _hunyuan_client

    if _hunyuan_client is None:
        if not hunyuan_space_id:
            logger.error("HUNYUAN_SPACE_ID is not provided")
            raise RuntimeError("HUNYUAN_SPACE_ID is not provided")
        try:
            logger.info(f"INFO: Initialising Gradio Client for space ID: {hunyuan_space_id}")
            _hunyuan_client = Client(hunyuan_space_id)
        except Exception as e:
            _hunyuan_client = None
            logger.error(f"ERROR: Failed to initialise Gradio Client for space ID {hunyuan_space_id}: {e}")
            raise RuntimeError(f"Failed to initialise Gradio Client: {e}") from e

    return _hunyuan_client


# Model handling
def save_generated_model(
    session_id: str,
    model_data: bytes,
    filename: str,
    sessions_dir: str,
    base_url: str
) -> Tuple[str, str]:
    """
    Saves the generated model binary data to the specified session directory

    Args:
        session_id (str): The unique identifier for the user session
        model_data (bytes): The binary data of the generated model
        filename (str): The desired filename for the saved model (e.g., 'model.glb')
        sessions_dir (str): The base directory for all sessions
        base_url (str): The base URL of the application for public URLs

    Raises:
        IOError: If there's an issue writing the model file

    Returns:
        Tuple[str, str]: A tuple containing the full file path of the saved model and its public URL
    """
    session_models_dir = os.path.join(sessions_dir, session_id, "models")

    os.makedirs(session_models_dir, exist_ok=True)
    file_path = os.path.join(session_models_dir, filename)

    try:
        with open(file_path, "wb") as f:
            f.write(model_data)
        logger.info(f"INFO: Saved model for session {session_id} to {file_path}")
    except IOError as e:
        logger.error(f"ERROR: Failed to write model data to {file_path}: {e}")
        raise IOError(f"Could not save model file: {e}") from e

    model_url = f"{base_url}/sessions/{session_id}/models/{filename}"
    return file_path, model_url


# Hunyuan API interaction
def _process_image_filepaths(
    image_filepaths: Dict[str, str], 
    predict_args: Dict,
    allowed_views: list
) -> Dict:
    """
    Prepares image file paths for the Gradio client's prediction

    Args:
        image_filepaths (Dict[str, str]): Dictionary mapping view names to file paths
        predict_args (Dict): Dictionary of prediction arguments to modify
        allowed_views (list): List of allowed view names

    Returns:
        Dict: Updated prediction arguments
        
    Raises:
        ValueError: If no image file paths provided or front image is missing
    """
    if not image_filepaths:
        raise ValueError("No image file paths provided for Hunyuan API call")

    front_image_path = image_filepaths.get("front")
    if not front_image_path:
        raise ValueError("Front image is required but not provided in image_filepaths")

    predict_args["image"] = handle_file(front_image_path)

    if len(image_filepaths) == 1:
        logger.info(f"INFO: Passing single image '{os.path.basename(front_image_path)}' to 'image' argument for Hunyuan API")
        return predict_args

    logger.info(f"INFO: Passing {len(image_filepaths)} images to multi-view arguments")

    for view in allowed_views:
        if view in image_filepaths and image_filepaths[view]:
            predict_args[f"mv_image_{view}"] = handle_file(image_filepaths[view])

    return predict_args


def call_hunyuan_shape_generation_api(
    image_filepaths: Dict[str, str],
    caption: str | None,
    hunyuan_space_id: str,
    hunyuan_api_name: str,
    allowed_views: list
) -> Tuple[bytes, str]:
    """
    Calls the Hunyuan shape generation API to process images and generate a 3D model

    Args:
        image_filepaths (Dict[str, str]): Dictionary mapping view names to file paths
        caption (Optional[str]): Optional caption for the model generation
        hunyuan_space_id (str): Hunyuan space identifier
        hunyuan_api_name (str): Name of the API endpoint to call
        allowed_views (list): List of allowed view names
        
    Returns:
        Tuple[bytes, str]: Binary model data and filename
        
    Raises:
        RuntimeError: If API name not provided or API call fails
        ValueError: If input validation fails
    """
    if not hunyuan_api_name:
        raise RuntimeError("HUNYUAN_API_NAME is not provided")

    try:
        client = get_hunyuan_client(hunyuan_space_id) # Pass space_id to client factory

        predict_args: Dict = {
            "caption": caption,
            "image": None,
            "mv_image_front": None,
            "mv_image_back": None,
            "mv_image_left": None,
            "mv_image_right": None,
            "api_name": hunyuan_api_name
        }

        predict_args = _process_image_filepaths(image_filepaths, predict_args, allowed_views)

        logger.info(f"INFO: Calling Hunyuan API '{hunyuan_api_name}' with images: {list(image_filepaths.keys())}")
        result = client.predict(**predict_args)

        generated_model_filepath = result[0]["value"]

        if not generated_model_filepath or not os.path.exists(generated_model_filepath):
            raise RuntimeError(f"Hunyuan API call failed or returned invalid model path: {generated_model_filepath}")

        with open(generated_model_filepath, "rb") as f:
            model_binary_data = f.read()

        model_filename = os.path.basename(generated_model_filepath)
        logger.info(f"INFO: Successfully received model '{model_filename}' from Hunyuan API")
        return model_binary_data, model_filename

    except ValueError as ve:
        logger.error(f"ERROR: Input validation failed for Hunyuan API call: {ve}")
        raise ve
    except Exception as e:
        logger.error(f"ERROR: Failed to call Hunyuan API for shape generation: {e}")
        raise RuntimeError(f"Failed to generate shape via Hunyuan API: {e}") from e
