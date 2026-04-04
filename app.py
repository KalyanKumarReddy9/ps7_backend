"""
Complete Backend for AI Image Detection
Fixed: Model loading, CORS, and error handling
"""

import os
import io
import sys
import time
import json
import traceback
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS
import gradio as gr
from PIL import Image
import numpy as np

# ============================================================================
# CONFIGURATION
# ============================================================================

APP_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_FILENAME = "final_inception_model.h5"
DEFAULT_PORT = 7860

# Global variables
model = None
model_load_error = None
model_load_status = "not_started"  # not_started, loading, loaded, failed
model_load_start_time = None
model_load_end_time = None

# TensorFlow lazy loading
tf = None

# ============================================================================
# FLASK APP INITIALIZATION
# ============================================================================

app = Flask(__name__)

# ============================================================================
# SIMPLIFIED CORS CONFIGURATION (PROVEN WORKING)
# ============================================================================

# Allow all origins with simple configuration
CORS(app, resources={r"/*": {"origins": "*"}})

# Add CORS headers manually as backup
@app.after_request
def add_cors_headers(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization, Accept'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    return response

# Handle preflight requests
@app.route('/predict', methods=['OPTIONS'])
def handle_options():
    response = jsonify({})
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    return response, 200

# ============================================================================
# TENSORFLOW LAZY LOADING
# ============================================================================

def get_tf():
    """Lazy load TensorFlow"""
    global tf
    if tf is None:
        print("[TensorFlow] Importing...")
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        import tensorflow as tf_module
        tf = tf_module
        print(f"[TensorFlow] Version: {tf.__version__}")
    return tf

# ============================================================================
# MODEL LOADING FUNCTIONS
# ============================================================================

def find_model_file():
    """Find model file"""
    # Check current directory
    if os.path.exists(MODEL_FILENAME):
        size = os.path.getsize(MODEL_FILENAME) / (1024 * 1024)
        print(f"[Model] Found: {MODEL_FILENAME} ({size:.2f} MB)")
        return MODEL_FILENAME
    
    # Check for any .h5 file
    for file in os.listdir('.'):
        if file.endswith('.h5'):
            size = os.path.getsize(file) / (1024 * 1024)
            print(f"[Model] Found alternative: {file} ({size:.2f} MB)")
            return file
    
    print(f"[Model] ERROR: No .h5 file found in {os.getcwd()}")
    print(f"[Model] Files present: {os.listdir('.')}")
    return None

def load_model_sync():
    """Load model synchronously with error handling"""
    global model, model_load_error, model_load_status, model_load_start_time, model_load_end_time
    
    model_load_status = "loading"
    model_load_start_time = time.time()
    
    print("=" * 60)
    print("[Model] Starting to load model...")
    print("=" * 60)
    
    try:
        # Find model file
        model_path = find_model_file()
        if not model_path:
            raise Exception(f"Model file '{MODEL_FILENAME}' not found")
        
        # Import TensorFlow
        tf_local = get_tf()
        
        # Load model
        print("[Model] Loading (this may take 1-2 minutes)...")
        start = time.time()
        loaded_model = tf_local.keras.models.load_model(model_path, compile=False)
        load_time = time.time() - start
        
        model = loaded_model
        model_load_status = "loaded"
        model_load_end_time = time.time()
        
        print(f"[Model] ✅ Loaded successfully in {load_time:.2f} seconds!")
        print("=" * 60)
        return True
        
    except Exception as e:
        model = None
        model_load_error = str(e)
        model_load_status = "failed"
        model_load_end_time = time.time()
        
        print(f"[Model] ❌ Failed to load: {e}")
        traceback.print_exc()
        print("=" * 60)
        return False

# ============================================================================
# START MODEL LOADING (SYNCHRONOUS - NO BACKGROUND THREAD)
# ============================================================================

print("=" * 60)
print("Starting Flask Application...")
print("=" * 60)

# Load model immediately (synchronously)
# This ensures model is loaded before any requests come in
load_model_sync()

# ============================================================================
# ROUTES
# ============================================================================

@app.route('/', methods=['GET'])
def root():
    return jsonify({
        "status": "online",
        "service": "AI Image Detection",
        "model_loaded": model is not None,
        "model_status": model_load_status,
        "endpoints": ["/health", "/model-status", "/debug", "/predict"]
    })

@app.route('/health', methods=['GET'])
def health():
    """Simple health check"""
    return jsonify({"status": "healthy", "timestamp": datetime.now().isoformat()}), 200

@app.route('/model-status', methods=['GET'])
def model_status():
    """Check model loading status"""
    return jsonify({
        "status": model_load_status,
        "model_loaded": model is not None,
        "model_file": MODEL_FILENAME,
        "error": model_load_error if model_load_status == "failed" else None,
        "load_time_seconds": round(model_load_end_time - model_load_start_time, 2) if model_load_end_time and model_load_start_time else None
    })

@app.route('/debug', methods=['GET'])
def debug():
    """Debug endpoint"""
    import psutil
    
    return jsonify({
        "working_directory": os.getcwd(),
        "files": os.listdir('.'),
        "model_status": model_load_status,
        "model_loaded": model is not None,
        "model_file_exists": os.path.exists(MODEL_FILENAME),
        "model_error": model_load_error,
        "memory_used_mb": psutil.Process().memory_info().rss / 1024 / 1024,
        "python_version": sys.version,
        "timestamp": datetime.now().isoformat()
    })

def preprocess_image(image, target_size=(299, 299)):
    """Preprocess image for InceptionV3"""
    tf_local = get_tf()
    
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    image = image.resize(target_size)
    img_array = np.array(image, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf_local.keras.applications.inception_v3.preprocess_input(img_array)
    
    return img_array

def predict_image(image):
    """Shared prediction helper for Flask and Gradio."""
    if model is None:
        return {
            "success": False,
            "error": "Model not loaded",
            "status": model_load_status,
            "message": model_load_error if model_load_error else "Model is not ready"
        }

    if image is None:
        return {"success": False, "error": "No image provided"}

    try:
        processed = preprocess_image(image)
        prediction = model.predict(processed, verbose=0)

        if len(prediction.shape) == 2 and prediction.shape[1] == 1:
            score = float(prediction[0][0])
            confidence = max(score, 1 - score)
            is_fake = score > 0.5

            return {
                "success": True,
                "prediction": "AI Generated" if is_fake else "Real Image",
                "confidence": round(confidence * 100, 2),
                "raw_score": score
            }

        return {
            "success": True,
            "prediction": prediction.tolist()
        }

    except Exception as e:
        print(f"Prediction error: {e}")
        traceback.print_exc()
        return {"success": False, "error": str(e)}

@app.route('/predict', methods=['POST', 'OPTIONS'])
def predict():
    """Make prediction"""
    if request.method == 'OPTIONS':
        return jsonify({}), 200

    # Check for image
    if 'image' not in request.files:
        return jsonify({"success": False, "error": "No image provided"}), 400
    
    file = request.files['image']
    if file.filename == '' or not file:
        return jsonify({"success": False, "error": "Invalid file"}), 400
    
    try:
        img_bytes = file.read()
        image = Image.open(io.BytesIO(img_bytes))
        result = predict_image(image)

        if not result.get("success"):
            return jsonify(result), 503 if result.get("error") == "Model not loaded" else 400

        return jsonify(result)

    except Exception as e:
        print(f"Prediction error: {e}")
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500

def build_gradio_app():
    """Create the Gradio interface for Hugging Face Spaces."""

    def gradio_predict(image):
        result = predict_image(image)
        if not result.get("success"):
            return result.get("error", "Prediction failed"), None, None, result

        prediction = result.get("prediction")
        confidence = result.get("confidence")
        raw_score = result.get("raw_score")

        if isinstance(prediction, list):
            prediction = json.dumps(prediction, indent=2)

        return prediction, confidence, raw_score, result

    with gr.Blocks(title="AI Image Detection") as demo:
        gr.Markdown(
            "# AI Image Detection\n"
            "Upload an image to check whether it is real or AI-generated."
        )
        with gr.Row():
            image_input = gr.Image(type="pil", label="Upload Image")
            with gr.Column():
                predict_button = gr.Button("Analyze Image", variant="primary")
                prediction_output = gr.Textbox(label="Prediction")
                confidence_output = gr.Number(label="Confidence (%)")
                raw_score_output = gr.Number(label="Raw Score")
                debug_output = gr.JSON(label="Full Result")

        predict_button.click(
            gradio_predict,
            inputs=[image_input],
            outputs=[prediction_output, confidence_output, raw_score_output, debug_output]
        )

    return demo

@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500

# ============================================================================
# RUN APPLICATION
# ============================================================================

if __name__ == '__main__':
    port = int(os.environ.get("PORT", DEFAULT_PORT))
    use_flask = os.environ.get("USE_FLASK", "0") == "1"

    print("=" * 60)
    print(f"Starting application on port {port}")
    print(f"Model status: {model_load_status}")
    print(f"Model loaded: {model is not None}")
    print(f"Mode: {'Flask' if use_flask else 'Gradio'}")
    print("=" * 60)

    if use_flask:
        app.run(host='0.0.0.0', port=port, threaded=True)
    else:
        demo = build_gradio_app()
        demo.launch(server_name='0.0.0.0', server_port=port, show_error=True)