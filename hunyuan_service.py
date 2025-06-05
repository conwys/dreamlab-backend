from gradio_client import Client, handle_file
from dotenv import load_dotenv
import os

load_dotenv()

hunyuan_space_id = os.getenv("HUNYUAN_SPACE_ID")
hunyuan_api_name = os.getenv("HUNYUAN_API_NAME")
sessions_dir = os.getenv("SESSIONS_DIR")

_hunyuan_client = None

def get_hunyuan_client():
    global _hunyuan_client

    if _hunyuan_client is None:
        try:
            _hunyuan_client = Client(hunyuan_space_id)
        except Exception as e:
            _hunyuan_client = None
            raise RuntimeError(f"Failed to init Gradio Client: {e}") from e
    
    return _hunyuan_client

def save_generated_model(session_id, model_data, filename):
    session_models_dir = os.path.join(sessions_dir, session_id, "models")
    file_path = os.path.join(session_models_dir, filename)

    with open(file_path, "wb") as f:
        f.write(model_data)

    base_url = "http://localhost:5000"
    model_url = f"{base_url}/sessions/{session_id}/models/{filename}"
    return file_path, model_url

def call_hunyuan_shape_generation_api(image_filepaths, caption):
    try:
        client = get_hunyuan_client()

        predict_args = {
            "caption": caption,
            "image": None,
            "mv_image_front": None,
            "mv_image_back": None,
            "mv_image_left": None,
            "mv_image_right": None,
            "api_name": hunyuan_api_name
        }

        predict_args = process_image_filepaths(image_filepaths, predict_args)
        
        result = client.predict(**predict_args)

        generated_model_filepath = result[0]["value"]
        
        if not generated_model_filepath or not os.path.exists(generated_model_filepath):
            raise RuntimeError("Hunyuan API call failed or returned no model file")
        
        with open(generated_model_filepath, "rb") as f:
            model_binary_data = f.read()

        model_filename = os.path.basename(generated_model_filepath)

        return model_binary_data, model_filename
    except Exception as e:
        raise RuntimeError(f"Failed to call Hunyuan API: {e}") from e

def process_image_filepaths(image_filepaths, predict_args):
    if not image_filepaths:
        raise ValueError("No image file paths provided to Hunyuan API call")

    front_image = image_filepaths.get("front")
    if not front_image:
        raise ValueError("Front image is required but not provided")

    predict_args["image"] = handle_file(front_image)

    if len(image_filepaths) == 1:
        print(f"INFO: Passing single image '{os.path.basename(front_image)}' to 'image' argument")
        return predict_args

    print(f"INFO: Passing {len(image_filepaths)} images to multi-view arguments")

    for view in ["front", "back", "left", "right"]:
        if view in image_filepaths:
            predict_args[f"mv_image_{view}"] = handle_file(image_filepaths[view])

    return predict_args
