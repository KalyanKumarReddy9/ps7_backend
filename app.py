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
CORS(app, resources={r"/*": {"origins": "*"}})

# Configuration
APP_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_FILENAME = "final_inception_model.h5"  # Single model file
MODEL_RETRY_SECONDS = 120  # 2 minutes for large model loading

# Model state
model = None
model_load_error = None
model_load_attempted = False
model_load_last_attempt_ts = 0
model_load_lock = threading.Lock()

def get_tf():
    """Lazy load TensorFlow to avoid startup delays"""
    global tf
    if tf is None:
        import tensorflow as tensorflow_module
        tf = tensorflow_module
        print("✅ TensorFlow imported successfully")
    return tf

def find_model_file():
    """Find the model file in various possible locations"""
    
    # Possible locations to check
    locations_to_check = [
        MODEL_FILENAME,  # Current directory
        os.path.join(APP_DIR, MODEL_FILENAME),  # App directory
        os.path.join(os.getcwd(), MODEL_FILENAME),  # Working directory
        "/opt/render/project/src/final_inception_model.h5",  # Render specific path
    ]
    
    for location in locations_to_check:
        if os.path.exists(location):
            file_size = os.path.getsize(location) / (1024 * 1024)  # Size in MB
            print(f"✅ Found model at: {location} ({file_size:.2f} MB)")
            return location
    
    print(f"❌ Model file '{MODEL_FILENAME}' not found in any location")
    print(f"Files in current directory: {os.listdir('.')}")
    return None

def load_model():
    """Load the model with simple error handling"""
    tf_local = get_tf()
    
    model_path = find_model_file()
    if not model_path:
        raise FileNotFoundError(f"Model file '{MODEL_FILENAME}' not found")
    
    print(f"Loading model from: {model_path}")
    print("This may take 30-60 seconds...")
    
    start_time = time.time()
    
    # Try different loading methods
    try:
        # Method 1: Standard load
        print("Attempt 1: Standard loading...")
        loaded_model = tf_local.keras.models.load_model(model_path, compile=False)
        print(f"✅ Model loaded successfully in {time.time() - start_time:.2f} seconds")
        return loaded_model
    except Exception as e1:
        print(f"Standard loading failed: {e1}")
        
        try:
            # Method 2: With safe_mode=False
            print("Attempt 2: Loading with safe_mode=False...")
            loaded_model = tf_local.keras.models.load_model(
                model_path, 
                compile=False, 
                safe_mode=False
            )
            print(f"✅ Model loaded successfully in {time.time() - start_time:.2f} seconds")
            return loaded_model
        except Exception as e2:
            print(f"Safe mode loading failed: {e2}")
            raise RuntimeError(f"Could not load model: {e1} | {e2}")

def ensure_model_loaded():
    """Ensure model is loaded with retry logic"""
    global model, model_load_error, model_load_attempted, model_load_last_attempt_ts
    
    # If already loaded, return True
    if model is not None:
        return True
    
    # Check if we should retry
    current_time = time.time()
    if model_load_attempted:
        time_since_attempt = current_time - model_load_last_attempt_ts
        if time_since_attempt < MODEL_RETRY_SECONDS:
            print(f"Model load attempted {time_since_attempt:.1f}s ago, waiting {MODEL_RETRY_SECONDS - time_since_attempt:.1f}s before retry")
            return False
    
    # Try to load the model
    with model_load_lock:
        # Double-check after acquiring lock
        if model is not None:
            return True
        
        model_load_attempted = True
        model_load_last_attempt_ts = current_time
        
        print("=" * 60)
        print("Starting model load process...")
        print(f"Working directory: {os.getcwd()}")
        print(f"App directory: {APP_DIR}")
        print(f"Files in directory: {os.listdir('.')}")
        print("=" * 60)
        
        try:
            model = load_model()
            model_load_error = None
            print("=" * 60)
            print("🎉 MODEL IS READY FOR PREDICTIONS!")
            print("=" * 60)
            return True
        except Exception as e:
            import traceback
            model = None
            model_load_error = str(e)
            print("=" * 60)
            print(f"❌ Failed to load model: {e}")
            print(traceback.format_exc())
            print("=" * 60)
            return False

# Start loading model in background
def warm_model_in_background():
    """Load model in background thread to not block startup"""
    time.sleep(2)  # Wait for Flask to start
    ensure_model_loaded()

# Start background thread
background_thread = threading.Thread(target=warm_model_in_background, daemon=True)
background_thread.start()

# ========== ROUTES ==========

@app.route('/', methods=['GET'])
def root():
    return jsonify({
        "status": "ok", 
        "service": "ps7-backend",
        "endpoints": ["/", "/health", "/model-health", "/loading-status", "/debug-paths", "/predict"]
    })

@app.route('/health', methods=['GET'])
def health():
    """Simple health check for Render"""
    return jsonify({"status": "ok", "service": "up"}), 200

@app.route('/loading-status', methods=['GET'])
def loading_status():
    """Check model loading progress"""
    global model, model_load_error, model_load_attempted
    
    if model is not None:
        return jsonify({
            "status": "ready",
            "model_loaded": True,
            "message": "Model is loaded and ready for predictions"
        })
    
    if model_load_error:
        return jsonify({
            "status": "failed",
            "model_loaded": False,
            "error": model_load_error
        }), 500
    
    if model_load_attempted:
        time_elapsed = time.time() - model_load_last_attempt_ts
        return jsonify({
            "status": "loading",
            "model_loaded": False,
            "message": f"Model loading in progress... ({time_elapsed:.0f}s elapsed)",
            "estimated_remaining": "30-60 seconds"
        })
    
    return jsonify({
        "status": "pending",
        "model_loaded": False,
        "message": "Model loading will start shortly"
    })

@app.route('/model-health', methods=['GET'])
def model_health():
    """Detailed model health check"""
    loaded = ensure_model_loaded()
    return jsonify({
        "status": "ready" if loaded else "error",
        "model_loaded": loaded,
        "model_file": MODEL_FILENAME,
        "model_load_error": model_load_error if not loaded else None
    }), 200 if loaded else 503

@app.route('/debug-paths', methods=['GET'])
def debug_paths():
    """Debug endpoint to check file locations"""
    import os
    
    debug_info = {
        "working_directory": os.getcwd(),
        "script_directory": APP_DIR,
        "model_filename": MODEL_FILENAME,
        "environment": {
            "MODEL_PATH": os.environ.get("MODEL_PATH", "Not set"),
            "PYTHONPATH": os.environ.get("PYTHONPATH", "Not set"),
            "PORT": os.environ.get("PORT", "Not set"),
        },
        "file_checks": {
            "model_exists_in_cwd": os.path.exists(MODEL_FILENAME),
            "model_exists_in_app_dir": os.path.exists(os.path.join(APP_DIR, MODEL_FILENAME)),
        }
    }
    
    # List all files in current directory
    try:
        all_files = os.listdir('.')
        debug_info["all_files_in_cwd"] = all_files
        
        # Filter for model files
        model_files = [f for f in all_files if any(ext in f.lower() for ext in ['.h5', '.keras', '.pb'])]
        debug_info["model_files_found"] = model_files
    except Exception as e:
        debug_info["error_listing_files"] = str(e)
    
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
    if request.method == 'OPTIONS':
        return jsonify({"status": "ok"}), 200
    
    # Check if model is loaded
    if not ensure_model_loaded():
        return jsonify({
            "error": "Model is still loading",
            "message": "Please wait a moment and try again",
            "status": "loading"
        }), 503
    
    # Check for image file
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "Empty file name"}), 400
    
    if not file:
        return jsonify({"error": "Empty file"}), 400
    
    try:
        # Read and preprocess image
        img_bytes = file.read()
        image = Image.open(io.BytesIO(img_bytes))
        processed_image = preprocess_image(image)
        
        # Make prediction
        prediction = model.predict(processed_image, verbose=0)
        
        # Handle different output shapes
        if len(prediction.shape) == 2 and prediction.shape[1] == 1:
            # Binary classification (sigmoid output)
            score = float(prediction[0][0])
            confidence = max(score, 1 - score)
            is_fake = score > 0.5  # Adjust threshold based on your model
            
            return jsonify({
                "success": True,
                "prediction": "AI Generated" if is_fake else "Real Image",
                "confidence": confidence,
                "raw_score": score,
                "threshold": 0.5
            })
        elif len(prediction.shape) == 2 and prediction.shape[1] == 2:
            # Binary classification (softmax output)
            predicted_class = int(np.argmax(prediction[0]))
            confidence = float(prediction[0][predicted_class])
            is_fake = predicted_class == 1  # Adjust based on your class ordering
            
            return jsonify({
                "success": True,
                "prediction": "AI Generated" if is_fake else "Real Image",
                "confidence": confidence,
                "class_probabilities": prediction[0].tolist()
            })
        else:
            # Unknown output format
            return jsonify({
                "success": True,
                "raw_prediction": prediction.tolist(),
                "shape": prediction.shape
            })
    
    except Exception as e:
        import traceback
        print(f"Prediction error: {e}")
        print(traceback.format_exc())
        return jsonify({
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
    print(f"Starting Flask app on port {port}")
    print(f"Model file: {MODEL_FILENAME}")
    print(f"Working directory: {os.getcwd()}")
    app.run(host='0.0.0.0', port=port, threaded=True)