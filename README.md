# DreamLab Backend

## Overview
The DreamLab Backend is a Flask-based REST API designed to handle session management, image uploads, and 3D model generation using the Hunyuan3D service. It provides endpoints for generating unique session IDs, processing furniture images, and retrieving generated 3D models.

## Features
- **Session Management**: Generate unique session IDs and create corresponding directories for uploads and models.
- **Image Upload**: Upload furniture images for processing.
- **3D Model Generation**: Integrates with the Hunyuan3D API to generate 3D models from uploaded images.
- **Model Retrieval**: Retrieve a list of generated 3D models for a session.

## Prerequisites
- Python 3.8 or higher
- Flask
- Gradio Client

## Installation
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd dreamlab-backend
   ```
2. (Recommended) Create and activate a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
1. Start the Flask server:
   ```bash
   python app.py
   ```
2. Access the API at `http://127.0.0.1:5000`.

## API Endpoints
### 1. Generate Session ID
**GET** `/api/generate_session_id`
- Generates a unique session ID and creates directories for uploads and models.
- **Response**:
  ```json
  {
    "session_id": "<unique-session-id>"
  }
  ```

### 2. Process Furniture Image
**POST** `/api/process_furniture_image/<string:session_id>`
- Uploads an image and processes it to generate a 3D model.
- **Request**:
  - Form-data: `image` (file)
- **Response**:
  ```json
  {
    "message": "Image saved successfully",
    "session_id": "<session-id>",
    "filename": "<uploaded-filename>"
  }
  ```

### 3. Retrieve Session Models
**GET** `/api/session_models/<string:session_id>`
- Retrieves a list of generated 3D models for the given session.
- **Response**:
  ```json
  {
    "session_id": "<session-id>",
    "models": ["model1.glb", "model2.glb"]
  }
  ```

## Directory Structure
```
./sessions/<session_id>/
  ├── uploads/   # Uploaded images
  ├── models/    # Generated 3D models
```

## Hunyuan3D Integration
The backend integrates with the Hunyuan3D API for 3D model generation. The `hunyuan_service.py` file contains the logic for interacting with the API.

## Acknowledgments
- [Flask](https://flask.palletsprojects.com/)
- [Gradio](https://gradio.app/)
- [Hunyuan3D](https://huggingface.co/spaces/tencent/Hunyuan3D-2)