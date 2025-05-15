from flask import Flask, jsonify, request
import uuid
import os

app = Flask(__name__)

# GET - Endpoint to generate a unique session ID
# Creates a directory for the session at ./sessions/<session_id>
@app.route("/api/generate_session_id", methods=["GET"])
def process_imgenerate_session_id():
    session_id = str(uuid.uuid4())
    os.makedirs(f"./sessions/{session_id}", exist_ok=True)
    os.makedirs(f"./sessions/{session_id}/uploads", exist_ok=True)
    os.makedirs(f"./sessions/{session_id}/models", exist_ok=True)
    return jsonify({
        "session_id": session_id
    }), 200

# POST - Endpoint to save image & call the hunyuan_service (which processes image and saves model) # So call your func??
# Uploads will be found at: ./sessions/<session_id>/uploads
@app.route("/api/process_furniture_image/<string:session_id>", methods=["POST"])
def process_image(session_id):
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    image = request.files['image']
    session_dir = f"./sessions/{session_id}/uploads"
    if not os.path.exists(session_dir):
        return jsonify({"error": "Session_id not found"}), 404
    image.save(os.path.join(session_dir, image.filename))

    #TODO:
    # Call the hunyuan_service to process the image into 3D model. Then save moodel(s) locally.
    # Only return 200 once complete.

    return jsonify({
        "message": "Image saved successfully",
        "session_id": session_id,
        "filename": image.filename
    }), 200


# GET - Endpoint to return a list of file names for models generated. 
# Models will be found at: ./sessions/<session_id>/models
@app.route("/api/session_models/<string:session_id>", methods=["GET"])
def get_session_models(session_id):
    session_dir = f"./sessions/{session_id}/models"
    if not os.path.exists(session_dir):
        return jsonify({"error": "Session_id not found"}), 404

    models = os.listdir(session_dir)
    return jsonify({
        "session_id": session_id,
        "models": models
    }), 200



if __name__ == "__main__":
    app.run(debug=True)