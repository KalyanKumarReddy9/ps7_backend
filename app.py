"""
Complete Backend for AI Image Detection
Handles: CORS, Model Loading, Predictions, Debugging
Author: Deployment Ready Version
"""

import os
import io
import sys
import time
import json
import threading
import traceback
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import numpy as np

# ============================================================================
# CONFIGURATION
# ============================================================================

APP_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_FILENAME = "final_inception_model.h5"
MODEL_LOAD_TIMEOUT = 300  # 5 minutes timeout for model loading

# Model state
model = None
model_load_error = None
model_load_status = "not_started"  # not_started, loading, loaded, failed
model_load_start_time = None
model_load_end_time = None
model_load_lock = threading.Lock()

# TensorFlow lazy loading
tf = None

# ============================================================================
# FLASK APP INITIALIZATION
# ============================================================================

app = Flask(__name__)

# ============================================================================
# COMPREHENSIVE CORS CONFIGURATION
# ============================================================================

# Method 1: Flask-CORS extension (primary)
CORS(app, 
     resources={r"/*": {
         "origins": "*",  # Allow all origins for production
         "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
         "allow_headers": ["Content-Type", "Authorization", "Accept", "Origin", "X-Requested-With"],
         "expose_headers": ["Content-Type", "Authorization"],
         "supports_credentials": True,
         "max_age": 3600
     }})

# Method 2: Manual CORS headers (backup - ensures CORS even if Flask-CORS fails)
@app.after_request
def add_cors_headers(response):
    """Add CORS headers to every response"""
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization, Accept, Origin, X-Requested-With'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, PUT, DELETE, OPTIONS, PATCH'
    response.headers['Access-Control-Allow-Credentials'] = 'true'
    response.headers['Access-Control-Max-Age'] = '3600'
    
    # Log CORS headers for debugging
    if request.method == 'OPTIONS':
        print(f"[CORS] Preflight request from {request.headers.get('Origin', 'unknown')}")
    
    return response

# Method 3: Explicit OPTIONS handler for all routes (catches preflight requests)
@app.before_request
def handle_preflight():
    """Handle CORS preflight requests explicitly"""
    if request.method == "OPTIONS":
        response = jsonify({"message": "CORS preflight successful"})
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization, Accept, Origin'
        response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
        return response, 200

# ============================================================================
# TENSORFLOW LAZY LOADING
# ============================================================================

def get_tf():
    """Lazy load TensorFlow to avoid startup delays"""
    global tf
    if tf is None:
        print("[TensorFlow] Importing TensorFlow module...")
        start_time = time.time()
        
        # Set TensorFlow logging level to reduce noise
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        
        import tensorflow as tf_module
        tf = tf_module
        
        # Log TensorFlow version
        print(f"[TensorFlow] Imported successfully in {time.time() - start_time:.2f}s")
        print(f"[TensorFlow] Version: {tf.__version__}")
        
        # Check if GPU is available
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"[TensorFlow] GPUs available: {len(gpus)}")
        else:
            print("[TensorFlow] No GPU detected, using CPU")
    
    return tf

# ============================================================================
# MODEL LOADING FUNCTIONS
# ============================================================================

def find_model_file():
    """Find model file in multiple locations with detailed logging"""
    print(f"[Model] Searching for model file: {MODEL_FILENAME}")
    
    # Locations to check
    locations = [
        MODEL_FILENAME,  # Current directory
        os.path.join(APP_DIR, MODEL_FILENAME),  # App directory
        os.path.join(os.getcwd(), MODEL_FILENAME),  # Working directory
        f"/opt/render/project/src/{MODEL_FILENAME}",  # Render specific path
    ]
    
    for location in locations:
        if os.path.exists(location):
            file_size = os.path.getsize(location) / (1024 * 1024)  # MB
            print(f"[Model] ✅ Found model at: {location}")
            print(f"[Model] File size: {file_size:.2f} MB")
            return location
    
    # If not found, list all files for debugging
    print(f"[Model] ❌ Model file not found!")
    print(f"[Model] Current directory: {os.getcwd()}")
    print(f"[Model] Files in current directory: {os.listdir('.')}")
    
    # Check for any .h5 or .keras file as fallback
    h5_files = [f for f in os.listdir('.') if f.endswith(('.h5', '.keras'))]
    if h5_files:
        print(f"[Model] Found alternative model files: {h5_files}")
        print(f"[Model] Using: {h5_files[0]}")
        return h5_files[0]
    
    return None

def load_model_with_retry():
    """Load model with multiple retry strategies and detailed logging"""
    tf_local = get_tf()
    
    model_path = find_model_file()
    if not model_path:
        raise FileNotFoundError(f"""
        Model file '{MODEL_FILENAME}' not found!
        
        Debug info:
        - Current directory: {os.getcwd()}
        - App directory: {APP_DIR}
        - Files present: {os.listdir('.')}
        - Expected file: {MODEL_FILENAME}
        
        Solution: Make sure the model file is uploaded to your repository.
        """)
    
    print(f"[Model] Loading model from: {model_path}")
    print(f"[Model] This may take 1-3 minutes depending on model size...")
    
    start_time = time.time()
    errors = []
    
    # Strategy 1: Standard loading
    try:
        print("[Model] Attempt 1/3: Standard loading...")
        loaded_model = tf_local.keras.models.load_model(model_path, compile=False)
        load_time = time.time() - start_time
        print(f"[Model] ✅ Model loaded successfully in {load_time:.2f}s")
        return loaded_model
    except Exception as e:
        error_msg = f"Standard loading failed: {str(e)[:200]}"
        print(f"[Model] ⚠️ {error_msg}")
        errors.append(error_msg)
    
    # Strategy 2: Load with safe_mode=False
    try:
        print("[Model] Attempt 2/3: Loading with safe_mode=False...")
        loaded_model = tf_local.keras.models.load_model(
            model_path, 
            compile=False, 
            safe_mode=False
        )
        load_time = time.time() - start_time
        print(f"[Model] ✅ Model loaded successfully in {load_time:.2f}s")
        return loaded_model
    except Exception as e:
        error_msg = f"Safe mode loading failed: {str(e)[:200]}"
        print(f"[Model] ⚠️ {error_msg}")
        errors.append(error_msg)
    
    # Strategy 3: Load with custom objects
    try:
        print("[Model] Attempt 3/3: Loading with custom_objects...")
        custom_objects = {
            'Functional': tf_local.keras.models.Model,
            'Sequential': tf_local.keras.models.Sequential,
            'DTypePolicy': tf_local.keras.mixed_precision.Policy,
            'Policy': tf_local.keras.mixed_precision.Policy
        }
        loaded_model = tf_local.keras.models.load_model(
            model_path,
            compile=False,
            custom_objects=custom_objects
        )
        load_time = time.time() - start_time
        print(f"[Model] ✅ Model loaded successfully in {load_time:.2f}s")
        return loaded_model
    except Exception as e:
        error_msg = f"Custom objects loading failed: {str(e)[:200]}"
        print(f"[Model] ⚠️ {error_msg}")
        errors.append(error_msg)
    
    # All strategies failed
    raise RuntimeError(f"""
    All model loading strategies failed!
    
    Errors encountered:
    {chr(10).join(f'  - {e}' for e in errors)}
    
    Possible solutions:
    1. Check if model file is corrupted
    2. Verify TensorFlow version compatibility
    3. Try re-saving the model locally
    4. Check available memory (Render free tier has 512MB)
    """)

def load_model_background():
    """Load model in background thread with status tracking"""
    global model, model_load_error, model_load_status, model_load_start_time, model_load_end_time
    
    with model_load_lock:
        if model_load_status in ["loading", "loaded"]:
            return
        
        model_load_status = "loading"
        model_load_start_time = time.time()
    
    print("=" * 70)
    print("[Model] Starting background model loading process...")
    print(f"[Model] Time: {datetime.now().isoformat()}")
    print(f"[Model] Python version: {sys.version}")
    print(f"[Model] Working directory: {os.getcwd()}")
    print("=" * 70)
    
    try:
        # Attempt to load the model
        loaded_model = load_model_with_retry()
        
        with model_load_lock:
            model = loaded_model
            model_load_error = None
            model_load_status = "loaded"
            model_load_end_time = time.time()
            
        load_duration = model_load_end_time - model_load_start_time
        print("=" * 70)
        print(f"[Model] 🎉 SUCCESS! Model loaded in {load_duration:.2f} seconds")
        print(f"[Model] Model is ready for predictions")
        print("=" * 70)
        
    except Exception as e:
        with model_load_lock:
            model = None
            model_load_error = str(e)
            model_load_status = "failed"
            model_load_end_time = time.time()
        
        print("=" * 70)
        print(f"[Model] ❌ FAILED to load model!")
        print(f"[Model] Error: {e}")
        print(f"[Model] Full traceback:")
        traceback.print_exc()
        print("=" * 70)

def ensure_model_loaded():
    """Ensure model is loaded (non-blocking)"""
    with model_load_lock:
        if model_load_status == "loaded":
            return True
        elif model_load_status == "not_started":
            # Start loading in background
            thread = threading.Thread(target=load_model_background, daemon=True)
            thread.start()
            return False
        elif model_load_status == "loading":
            return False
        else:  # failed
            return False

# ============================================================================
# HEALTH AND DEBUG ENDPOINTS
# ============================================================================

@app.route('/', methods=['GET', 'OPTIONS'])
def root():
    """Root endpoint - API information"""
    if request.method == 'OPTIONS':
        return jsonify({}), 200
    
    return jsonify({
        "status": "online",
        "service": "AI Image Detection Backend",
        "version": "2.0.0",
        "timestamp": datetime.now().isoformat(),
        "model_status": model_load_status,
        "endpoints": {
            "health": "/health",
            "model_health": "/model-health",
            "debug": "/debug",
            "predict": "/predict (POST)",
            "cors_test": "/cors-test"
        },
        "documentation": {
            "frontend_repo": "https://github.com/your-repo",
            "api_base_url": "https://ps7-backend-fhze.onrender.com"
        }
    })

@app.route('/health', methods=['GET', 'OPTIONS'])
def health():
    """Simple health check for Render"""
    if request.method == 'OPTIONS':
        return jsonify({}), 200
    
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "uptime_seconds": time.time() - app_start_time if 'app_start_time' in globals() else 0
    }), 200

@app.route('/model-health', methods=['GET', 'OPTIONS'])
def model_health():
    """Detailed model health check"""
    if request.method == 'OPTIONS':
        return jsonify({}), 200
    
    with model_load_lock:
        response = {
            "status": model_load_status,
            "model_loaded": model is not None,
            "model_file": MODEL_FILENAME,
            "timestamp": datetime.now().isoformat()
        }
        
        if model_load_status == "loaded":
            response["message"] = "Model is ready for predictions"
            if model_load_start_time and model_load_end_time:
                response["load_duration_seconds"] = round(model_load_end_time - model_load_start_time, 2)
        
        elif model_load_status == "loading":
            if model_load_start_time:
                elapsed = time.time() - model_load_start_time
                response["message"] = f"Model loading in progress ({elapsed:.0f}s elapsed)"
                response["elapsed_seconds"] = round(elapsed, 2)
                response["estimated_remaining"] = "1-2 minutes"
        
        elif model_load_status == "failed":
            response["message"] = "Model failed to load"
            response["error"] = model_load_error
            response["troubleshooting"] = [
                "Check Render logs for detailed error",
                "Verify model file exists and is not corrupted",
                "Ensure sufficient memory (Render free tier has 512MB)",
                "Try converting model to TensorFlow Lite format"
            ]
        
        else:  # not_started
            response["message"] = "Model loading not yet started"
        
        http_status = 200 if model_load_status == "loaded" else 503 if model_load_status == "loading" else 500
        return jsonify(response), http_status

@app.route('/debug', methods=['GET', 'OPTIONS'])
def debug_info():
    """Comprehensive debug endpoint - use for troubleshooting"""
    if request.method == 'OPTIONS':
        return jsonify({}), 200
    
    import psutil
    
    debug_data = {
        "timestamp": datetime.now().isoformat(),
        "system_info": {
            "working_directory": os.getcwd(),
            "app_directory": APP_DIR,
            "python_version": sys.version,
            "platform": sys.platform,
        },
        "model_info": {
            "status": model_load_status,
            "model_loaded": model is not None,
            "model_filename": MODEL_FILENAME,
            "model_file_exists": os.path.exists(MODEL_FILENAME),
            "model_load_error": model_load_error if model_load_status == "failed" else None,
        },
        "memory_info": {
            "current_process_mb": psutil.Process().memory_info().rss / 1024 / 1024,
            "system_total_mb": psutil.virtual_memory().total / 1024 / 1024,
            "system_available_mb": psutil.virtual_memory().available / 1024 / 1024,
            "system_percent_used": psutil.virtual_memory().percent,
        },
        "files_in_directory": os.listdir('.'),
        "environment_variables": {
            "PORT": os.environ.get("PORT", "Not set"),
            "MODEL_PATH": os.environ.get("MODEL_PATH", "Not set"),
            "CORS_ORIGINS": os.environ.get("CORS_ORIGINS", "Not set - using '*'"),
        }
    }
    
    # Add model file size if exists
    if os.path.exists(MODEL_FILENAME):
        debug_data["model_info"]["model_size_mb"] = round(
            os.path.getsize(MODEL_FILENAME) / (1024 * 1024), 2
        )
    
    # Add TensorFlow info if loaded
    if tf is not None:
        debug_data["tensorflow_info"] = {
            "version": tf.__version__,
            "gpu_available": len(tf.config.list_physical_devices('GPU')) > 0,
        }
    
    return jsonify(debug_data)

@app.route('/cors-test', methods=['GET', 'OPTIONS'])
def cors_test():
    """Endpoint to test CORS configuration"""
    if request.method == 'OPTIONS':
        response = jsonify({})
        response.headers['Access-Control-Allow-Origin'] = '*'
        return response, 200
    
    return jsonify({
        "cors_status": "working",
        "message": "CORS is properly configured",
        "headers_received": dict(request.headers),
        "origin": request.headers.get('Origin', 'No origin header'),
        "method": request.method
    })

# ============================================================================
# PREDICTION ENDPOINT
# ============================================================================

def preprocess_image(image, target_size=(299, 299)):
    """Preprocess image for InceptionV3 model"""
    tf_local = get_tf()
    
    # Convert to RGB if needed
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    # Resize
    image = image.resize(target_size)
    
    # Convert to array
    img_array = np.array(image, dtype=np.float32)
    
    # Expand dimensions to create batch
    img_array = np.expand_dims(img_array, axis=0)
    
    # InceptionV3 preprocessing (scales to [-1, 1])
    img_array = tf_local.keras.applications.inception_v3.preprocess_input(img_array)
    
    return img_array

@app.route('/predict', methods=['POST', 'OPTIONS'])
def predict():
    """
    Make prediction on uploaded image
    
    Expected: multipart/form-data with field 'image'
    Returns: Prediction result with confidence score
    """
    # Handle preflight
    if request.method == 'OPTIONS':
        response = jsonify({})
        response.headers['Access-Control-Allow-Origin'] = '*'
        return response, 200
    
    # Log request
    print(f"[Predict] Request from {request.headers.get('Origin', 'unknown')}")
    print(f"[Predict] Content-Type: {request.content_type}")
    
    # Check model status
    if model_load_status != "loaded":
        status_msg = {
            "not_started": "Model loading not started yet",
            "loading": f"Model is loading ({(time.time() - model_load_start_time):.0f}s elapsed)",
            "failed": f"Model failed to load: {model_load_error}",
        }.get(model_load_status, "Unknown model status")
        
        return jsonify({
            "success": False,
            "error": "Model not ready",
            "message": status_msg,
            "model_status": model_load_status
        }), 503
    
    # Check for image file
    if 'image' not in request.files:
        return jsonify({
            "success": False,
            "error": "No image provided",
            "message": "Please send an image file with field name 'image'"
        }), 400
    
    file = request.files['image']
    if file.filename == '' or not file:
        return jsonify({
            "success": False,
            "error": "Invalid file",
            "message": "File is empty or has no name"
        }), 400
    
    try:
        # Read and preprocess image
        print(f"[Predict] Processing image: {file.filename}")
        img_bytes = file.read()
        image = Image.open(io.BytesIO(img_bytes))
        processed_image = preprocess_image(image)
        
        # Make prediction
        print("[Predict] Running inference...")
        start_time = time.time()
        prediction = model.predict(processed_image, verbose=0)
        inference_time = time.time() - start_time
        print(f"[Predict] Inference completed in {inference_time:.2f}s")
        
        # Format response based on output shape
        if len(prediction.shape) == 2 and prediction.shape[1] == 1:
            # Binary classification (sigmoid output)
            score = float(prediction[0][0])
            confidence = max(score, 1 - score)
            is_fake = score > 0.5
            
            response = {
                "success": True,
                "prediction": "AI Generated" if is_fake else "Real Image",
                "confidence": round(confidence * 100, 2),
                "raw_score": score,
                "inference_time_ms": round(inference_time * 1000, 2)
            }
            
            print(f"[Predict] Result: {response['prediction']} (confidence: {response['confidence']}%)")
            return jsonify(response)
        
        elif len(prediction.shape) == 2 and prediction.shape[1] == 2:
            # Binary classification (softmax output)
            predicted_class = int(np.argmax(prediction[0]))
            confidence = float(prediction[0][predicted_class])
            is_fake = predicted_class == 1
            
            response = {
                "success": True,
                "prediction": "AI Generated" if is_fake else "Real Image",
                "confidence": round(confidence * 100, 2),
                "class_probabilities": {
                    "real": float(prediction[0][0]),
                    "fake": float(prediction[0][1])
                },
                "inference_time_ms": round(inference_time * 1000, 2)
            }
            
            print(f"[Predict] Result: {response['prediction']} (confidence: {response['confidence']}%)")
            return jsonify(response)
        
        else:
            # Unknown output format
            return jsonify({
                "success": True,
                "raw_prediction": prediction.tolist(),
                "shape": prediction.shape,
                "inference_time_ms": round(inference_time * 1000, 2)
            })
    
    except Exception as e:
        print(f"[Predict] ❌ Error: {e}")
        traceback.print_exc()
        return jsonify({
            "success": False,
            "error": "Prediction failed",
            "message": str(e),
            "type": type(e).__name__
        }), 500

# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({
        "error": "Not Found",
        "message": "The requested endpoint does not exist",
        "available_endpoints": [
            "/",
            "/health",
            "/model-health", 
            "/debug",
            "/cors-test",
            "/predict"
        ]
    }), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    print(f"[Error] 500 Internal Server Error: {error}")
    return jsonify({
        "error": "Internal Server Error",
        "message": "Something went wrong on the server",
        "timestamp": datetime.now().isoformat()
    }), 500

# ============================================================================
# APPLICATION STARTUP
# ============================================================================

app_start_time = time.time()

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    
    print("=" * 70)
    print("🚀 AI IMAGE DETECTION BACKEND")
    print("=" * 70)
    print(f"Server starting at: {datetime.now().isoformat()}")
    print(f"Port: {port}")
    print(f"Working directory: {os.getcwd()}")
    print(f"Model file: {MODEL_FILENAME}")
    print(f"Model status: {model_load_status}")
    print("=" * 70)
    print("CORS Configuration:")
    print("  - All origins allowed: *")
    print("  - Methods: GET, POST, OPTIONS")
    print("  - Headers: Content-Type, Authorization, Accept")
    print("=" * 70)
    print("Available endpoints:")
    print("  GET  /            - API information")
    print("  GET  /health      - Health check")
    print("  GET  /model-health - Model status")
    print("  GET  /debug       - Debug information")
    print("  GET  /cors-test   - Test CORS")
    print("  POST /predict     - Make prediction")
    print("=" * 70)
    
    # Start model loading in background
    ensure_model_loaded()
    
    # Run Flask app
    app.run(host='0.0.0.0', port=port, threaded=True)