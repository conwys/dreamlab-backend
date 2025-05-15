from gradio_client import Client, handle_file
import os
import time

HUNYUAN_SPACE_ID = "tencent/Hunyuan3D-2"
HUNYUAN_API_NAME = "/shape_generation"
SESSIONS_DIR = "./sessions"

_hunyuan_client = None

def get_hunyuan_client():
    global _hunyuan_client

    if _hunyuan_client is None:
        try:
            _hunyuan_client = Client(HUNYUAN_SPACE_ID)
        except Exception as e:
            _hunyuan_client = False
            raise RuntimeError(f"Failed to init Gradio Client: {e}") from e
        
    if _hunyuan_client is False:
        raise RuntimeError("Gradio Client failed to init previously")
    
    return _hunyuan_client

def save_generated_model(session_id, model_data, filename):
    session_models_dir = f"./sessions/{session_id}/models"
    file_path = os.path.join(session_models_dir, filename)

    # Write 3d model data to file
    with open(file_path, "wb") as f:
        f.write(model_data)

    # Construct a url string that the frontend can use to request this specific file
    base_url = "http://localhost:5000"  # TODO: Replace with your actual base URL
    model_url = f"{base_url}/sessions/{session_id}/models/{filename}"
    return file_path, model_url

def call_hunyuan_shape_generation_api(image_filepath: str, caption: str = None):
    try:
        client = get_hunyuan_client()

        result = client.predict(
            caption=caption,
            image=image_filepath,
            steps=5,
            guidance_scale=5,
            seed=1234,
            octree_resolution=256,
            check_box_rembg=True,
            num_chunks=8000,
            randomize_seed=False,
            api_name=HUNYUAN_API_NAME
        )
    except Exception as e:
        raise RuntimeError(f"Failed to call Hunyuan API: {e}") from e