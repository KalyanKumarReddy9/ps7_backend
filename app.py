import os
import io
import threading
import time
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image

# Global variables
tf = None
app = Flask(__name__)

# ========== CORS CONFIGURATION - FULLY FIXED ==========
# Allow all origins for production
CORS(app, 
     resources={r"/*": {"origins": "*"}},
     methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
     allow_headers=["Content-Type", "Authorization", "Accept", "Origin"],
     expose_headers=["Content-Type", "Authorization"],
     supports_credentials=True,
     max_age=3600)

# Force CORS headers on EVERY response
@app.after_request
def add_cors_headers(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization, Accept, Origin'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, PUT, DELETE, OPTIONS'
    response.headers['Access-Control-Allow-Credentials'] = 'true'
    response.headers['Access-Control-Max-Age'] = '3600'
    return response

# Handle preflight requests for all routes
@app.before_request
def handle_preflight():
    if request.method == "OPTIONS":
        response = jsonify({})
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization, Accept, Origin'
        response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
        return response, 200

# ========== CONFIGURATION ==========
APP_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_FILENAME = "final_inception_model.h5"

# Model state
model = None
model_load_error = None
model_load_attempted = False
model_load_start_time = None
model_load_lock = threading.Lock()

def get_tf():
    """Lazy load TensorFlow to avoid startup delays"""
    global tf
    if tf is None:
        print("🔄 Importing TensorFlow (this may take 30-60 seconds)...")
        import tensorflow as tensorflow_module
        tf = tensorflow_module
        print(f"✅ TensorFlow {tf.__version__} imported successfully")
    return tf

def find_model_file():
    """Find the model file in various possible locations"""
    
    # List all files for debugging
    print(f"Searching for {MODEL_FILENAME}...")
    current_files = os.listdir('.')
    print(f"Current directory contents: {current_files}")
    
    # Check if model exists in current directory
    if os.path.exists(MODEL_FILENAME):
        file_size = os.path.getsize(MODEL_FILENAME) / (1024 * 1024)
        print(f"✅ Found model at: {MODEL_FILENAME} ({file_size:.2f} MB)")
        return MODEL_FILENAME
    
    # Check app directory
    app_dir_path = os.path.join(APP_DIR, MODEL_FILENAME)
    if os.path.exists(app_dir_path):
        file_size = os.path.getsize(app_dir_path) / (1024 * 1024)
        print(f"✅ Found model at: {app_dir_path} ({file_size:.2f} MB)")
        return app_dir_path
    
    # Check for any .h5 file as fallback
    h5_files = [f for f in current_files if f.endswith('.h5')]
    if h5_files:
        print(f"⚠️ Found alternative .h5 file: {h5_files[0]}")
        return h5_files[0]
    
    print(f"❌ Model file '{MODEL_FILENAME}' not found")
    return None

def load_model():
    """Load the model with comprehensive error handling"""
    tf_local = get_tf()
    
    model_path = find_model_file()
    if not model_path:
        raise FileNotFoundError(f"Model file '{MODEL_FILENAME}' not found in {os.getcwd()}")
    
    print(f"Loading model from: {model_path}")
    print("This may take 1-2 minutes depending on model size...")
    
    start_time = time.time()
    
    # Try multiple loading methods
    errors = []
    
    # Method 1: Standard load
    try:
        print("Attempt 1: Standard loading...")
        loaded_model = tf_local.keras.models.load_model(model_path, compile=False)
        load_time = time.time() - start_time
        print(f"✅ Model loaded successfully in {load_time:.2f} seconds")
        return loaded_model
    except Exception as e:
        error_msg = f"Standard loading failed: {str(e)[:100]}"
        print(error_msg)
        errors.append(error_msg)
    
    # Method 2: With safe_mode=False
    try:
        print("Attempt 2: Loading with safe_mode=False...")
        loaded_model = tf_local.keras.models.load_model(
            model_path, 
            compile=False, 
            safe_mode=False
        )
        load_time = time.time() - start_time
        print(f"✅ Model loaded successfully in {load_time:.2f} seconds")
        return loaded_model
    except Exception as e:
        error_msg = f"Safe mode loading failed: {str(e)[:100]}"
        print(error_msg)
        errors.append(error_msg)
    
    # Method 3: Try with custom_objects for compatibility
    try:
        print("Attempt 3: Loading with custom_objects...")
        custom_objects = {
            'Functional': tf_local.keras.models.Model,
            'Sequential': tf_local.keras.models.Sequential
        }
        loaded_model = tf_local.keras.models.load_model(
            model_path,
            compile=False,
            custom_objects=custom_objects
        )
        load_time = time.time() - start_time
        print(f"✅ Model loaded successfully in {load_time:.2f} seconds")
        return loaded_model
    except Exception as e:
        error_msg = f"Custom objects loading failed: {str(e)[:100]}"
        print(error_msg)
        errors.append(error_msg)
    
    # All methods failed
    raise RuntimeError(f"All loading methods failed: {' | '.join(errors)}")

def ensure_model_loaded():
    """Ensure model is loaded with retry logic"""
    global model, model_load_error, model_load_attempted, model_load_start_time
    
    # If already loaded, return True
    if model is not None:
        return True
    
    # If loading hasn't started, start it
    if not model_load_attempted:
        with model_load_lock:
            if not model_load_attempted:
                model_load_attempted = True
                model_load_start_time = time.time()
                
                print("=" * 60)
                print("Starting model load process...")
                print(f"Working directory: {os.getcwd()}")
                print(f"Python version: {os.sys.version}")
                print("=" * 60)
                
                # Start loading in background
                def load_in_background():
                    global model, model_load_error
                    try:
                        model = load_model()
                        model_load_error = None
                        print("=" * 60)
                        print("🎉 MODEL IS READY FOR PREDICTIONS!")
                        print("=" * 60)
                    except Exception as e:
                        import traceback
                        model = None
                        model_load_error = str(e)
                        print("=" * 60)
                        print(f"❌ Failed to load model: {e}")
                        print(traceback.format_exc())
                        print("=" * 60)
                
                thread = threading.Thread(target=load_in_background, daemon=True)
                thread.start()
                return False
    
    # Loading is in progress or failed
    return False

# Start model loading immediately on startup
print("🚀 Starting application...")
ensure_model_loaded()

# ========== ROUTES ==========

@app.route('/', methods=['GET', 'OPTIONS'])
def root():
    if request.method == 'OPTIONS':
        return jsonify({}), 200
    
    return jsonify({
        "status": "ok", 
        "service": "ps7-backend",
        "version": "2.0.0",
        "model_loaded": model is not None,
        "endpoints": [
            "/",
            "/health",
            "/model-health", 
            "/loading-status", 
            "/debug-paths", 
            "/predict"
        ]
    })

@app.route('/health', methods=['GET', 'OPTIONS'])
def health():
    if request.method == 'OPTIONS':
        return jsonify({}), 200
    return jsonify({"status": "ok", "service": "up", "timestamp": time.time()}), 200

@app.route('/loading-status', methods=['GET', 'OPTIONS'])
def loading_status():
    """Check model loading progress"""
    if request.method == 'OPTIONS':
        return jsonify({}), 200
    
    if model is not None:
        return jsonify({
            "status": "ready",
            "model_loaded": True,
            "message": "Model is ready for predictions",
            "load_time": None if model_load_start_time is None else time.time() - model_load_start_time
        })
    
    if model_load_error:
        return jsonify({
            "status": "failed",
            "model_loaded": False,
            "error": model_load_error,
            "message": "Model failed to load. Check logs for details."
        }), 500
    
    if model_load_attempted and model_load_start_time:
        elapsed = time.time() - model_load_start_time
        return jsonify({
            "status": "loading",
            "model_loaded": False,
            "message": f"Model loading in progress... ({elapsed:.0f}s elapsed)",
            "estimated_time": "1-2 minutes"
        })
    
    return jsonify({
        "status": "pending",
        "model_loaded": False,
        "message": "Model loading will start shortly"
    })

@app.route('/model-health', methods=['GET', 'OPTIONS'])
def model_health():
    """Detailed model health check"""
    if request.method == 'OPTIONS':
        return jsonify({}), 200
    
    is_loaded = model is not None
    
    return jsonify({
        "status": "ready" if is_loaded else "loading",
        "model_loaded": is_loaded,
        "model_file": MODEL_FILENAME,
        "model_load_error": model_load_error if not is_loaded else None,
        "timestamp": time.time()
    }), 200 if is_loaded else 202

@app.route('/debug-paths', methods=['GET', 'OPTIONS'])
def debug_paths():
    """Comprehensive debug endpoint"""
    if request.method == 'OPTIONS':
        return jsonify({}), 200
    
    import sys
    
    debug_info = {
        "working_directory": os.getcwd(),
        "app_directory": APP_DIR,
        "model_filename": MODEL_FILENAME,
        "model_exists": os.path.exists(MODEL_FILENAME),
        "model_loaded": model is not None,
        "model_load_error": model_load_error,
        "all_files": os.listdir('.'),
        "python_version": sys.version,
        "environment": {
            "PORT": os.environ.get("PORT", "Not set"),
            "MODEL_PATH": os.environ.get("MODEL_PATH", "Not set"),
        }
    }
    
    # Add model file size if exists
    if os.path.exists(MODEL_FILENAME):
        debug_info["model_size_mb"] = round(os.path.getsize(MODEL_FILENAME) / (1024 * 1024), 2)
    
    # Add TensorFlow info if imported
    if tf is not None:
        debug_info["tensorflow_version"] = tf.__version__
    else:
        debug_info["tensorflow_version"] = "Not imported yet"
    
    return jsonify(debug_info)

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
    """Make prediction on uploaded image"""
    # Handle preflight
    if request.method == 'OPTIONS':
        response = jsonify({})
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization, Accept'
        response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
        return response, 200
    
    # Check if model is loaded
    if model is None:
        # Try to ensure model is loaded
        if not ensure_model_loaded():
            return jsonify({
                "success": False,
                "error": "Model is still loading",
                "message": "Please wait 1-2 minutes for the model to load",
                "status": "loading"
            }), 503
    
    # Double-check model after potential load
    if model is None:
        return jsonify({
            "success": False,
            "error": "Model failed to load",
            "message": model_load_error or "Unknown error loading model",
            "status": "error"
        }), 500
    
    # Check for image file
    if 'image' not in request.files:
        return jsonify({
            "success": False,
            "error": "No image provided"
        }), 400
    
    file = request.files['image']
    if file.filename == '' or not file:
        return jsonify({
            "success": False,
            "error": "Invalid file"
        }), 400
    
    try:
        # Read and preprocess image
        img_bytes = file.read()
        image = Image.open(io.BytesIO(img_bytes))
        processed_image = preprocess_image(image)
        
        # Make prediction
        prediction = model.predict(processed_image, verbose=0)
        
        # Format response based on output shape
        if len(prediction.shape) == 2:
            if prediction.shape[1] == 1:
                # Binary classification (sigmoid output)
                score = float(prediction[0][0])
                confidence = max(score, 1 - score)
                is_fake = score > 0.5
                
                return jsonify({
                    "success": True,
                    "prediction": "AI Generated" if is_fake else "Real Image",
                    "confidence": round(confidence * 100, 2),
                    "raw_score": score,
                    "threshold": 0.5
                })
            elif prediction.shape[1] == 2:
                # Binary classification (softmax output)
                predicted_class = int(np.argmax(prediction[0]))
                confidence = float(prediction[0][predicted_class])
                is_fake = predicted_class == 1
                
                return jsonify({
                    "success": True,
                    "prediction": "AI Generated" if is_fake else "Real Image",
                    "confidence": round(confidence * 100, 2),
                    "class_probabilities": {
                        "real": float(prediction[0][0]),
                        "fake": float(prediction[0][1])
                    }
                })
        
        # Fallback for unknown format
        return jsonify({
            "success": True,
            "prediction": prediction.tolist(),
            "shape": prediction.shape
        })
    
    except Exception as e:
        import traceback
        print(f"Prediction error: {e}")
        print(traceback.format_exc())
        return jsonify({
            "success": False,
            "error": "Prediction failed",
            "details": str(e)
        }), 500

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({
        "error": "Not Found",
        "message": "The requested endpoint does not exist",
        "available_endpoints": ["/", "/health", "/loading-status", "/model-health", "/debug-paths", "/predict"]
    }), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        "error": "Internal Server Error",
        "message": "Something went wrong on the server"
    }), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    print("=" * 60)
    print(f"Starting Flask app on port {port}")
    print(f"Model file: {MODEL_FILENAME}")
    print(f"Working directory: {os.getcwd()}")
    print("CORS is configured to allow all origins")
    print("=" * 60)
    app.run(host='0.0.0.0', port=port, threaded=True)