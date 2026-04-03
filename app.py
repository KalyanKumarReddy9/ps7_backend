import os
import io
import json
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
import h5py
from PIL import Image

tf = None


def get_tf():
    global tf
    if tf is None:
        import tensorflow as tensorflow_module
        tf = tensorflow_module
    return tf


def load_model_with_compatibility(model_path: str):
    """Load model with compatibility fallbacks for common Keras serialization mismatches."""
    tf_local = get_tf()
    custom_objects = {
        # Newer saved configs may include this class name while older runtimes expect Policy.
        "DTypePolicy": tf_local.keras.mixed_precision.Policy,
    }

    attempts = []

    def _attempt_default():
        return tf_local.keras.models.load_model(model_path, compile=False)

    def _attempt_custom_objects():
        return tf_local.keras.models.load_model(
            model_path,
            compile=False,
            custom_objects=custom_objects,
        )

    def _attempt_inputlayer_patch(use_custom_objects: bool):
        original_init = tf_local.keras.layers.InputLayer.__init__

        def patched_init(self, *args, **kwargs):
            if "batch_shape" in kwargs and "batch_input_shape" not in kwargs:
                kwargs["batch_input_shape"] = kwargs.pop("batch_shape")
            return original_init(self, *args, **kwargs)

        tf_local.keras.layers.InputLayer.__init__ = patched_init
        try:
            if use_custom_objects:
                return tf_local.keras.models.load_model(
                    model_path,
                    compile=False,
                    custom_objects=custom_objects,
                )
            return tf_local.keras.models.load_model(model_path, compile=False)
        finally:
            tf_local.keras.layers.InputLayer.__init__ = original_init

    loaders = [
        _attempt_default,
        _attempt_custom_objects,
        lambda: _attempt_inputlayer_patch(use_custom_objects=False),
        lambda: _attempt_inputlayer_patch(use_custom_objects=True),
        lambda: _attempt_h5_config_rewrite(model_path, custom_objects),
    ]

    for loader in loaders:
        try:
            return loader()
        except Exception as err:
            attempts.append(str(err))

    raise RuntimeError(" | ".join(attempts))


def _patch_model_config(obj):
    if isinstance(obj, dict):
        if obj.get("class_name") == "DTypePolicy":
            obj["class_name"] = "Policy"
            obj["module"] = "keras.mixed_precision"

        config = obj.get("config")
        if isinstance(config, dict) and "batch_shape" in config and "batch_input_shape" not in config:
            config["batch_input_shape"] = config.pop("batch_shape")

        for key, value in list(obj.items()):
            obj[key] = _patch_model_config(value)
        return obj

    if isinstance(obj, list):
        return [_patch_model_config(item) for item in obj]

    return obj


def _attempt_h5_config_rewrite(model_path: str, custom_objects: dict):
    tf_local = get_tf()
    with h5py.File(model_path, "r") as h5_file:
        raw_model_config = h5_file.attrs.get("model_config")

    if raw_model_config is None:
        raise RuntimeError("H5 file does not contain model_config")

    if isinstance(raw_model_config, bytes):
        raw_model_config = raw_model_config.decode("utf-8")

    parsed_config = json.loads(raw_model_config)
    patched_config = _patch_model_config(parsed_config)

    rebuilt_model = tf_local.keras.models.model_from_json(
        json.dumps(patched_config),
        custom_objects=custom_objects,
    )
    rebuilt_model.load_weights(model_path)
    return rebuilt_model

app = Flask(__name__)
# Enable CORS for the React frontend
CORS(app)

model = None
model_load_error = None
model_load_attempted = False


def ensure_model_loaded():
    global model, model_load_error, model_load_attempted
    if model is not None:
        return True
    if model_load_attempted and model is None:
        return False

    model_load_attempted = True
    try:
        model = load_model_with_compatibility('final_inception_model.h5')
        model_load_error = None
        print("Model loaded successfully!")
        return True
    except Exception as e:
        print("Error loading model:", e)
        model = None
        model_load_error = str(e)
        return False


@app.route('/', methods=['GET'])
def root():
    return jsonify({"status": "ok", "service": "ps7-backend"}), 200


@app.route('/health', methods=['GET'])
def health():
    # Keep this endpoint lightweight for Render health probes.
    return jsonify({"status": "ok", "service": "up"}), 200


@app.route('/model-health', methods=['GET'])
def model_health():
    loaded = ensure_model_loaded()
    status_code = 200 if loaded else 500
    return jsonify({
        "status": "ok" if loaded else "error",
        "model_loaded": loaded,
        "model_load_error": model_load_error
    }), status_code

def preprocess_image(image, target_size=(299, 299)):
    tf_local = get_tf()
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(target_size)
    img_array = np.array(image)
    
    # Expand dimensions to create a batch size of 1
    img_array = np.expand_dims(img_array, axis=0)

    # InceptionV3 preprocessing
    # Or simply: img_array = img_array / 255.0 depending on how the model was trained.
    img_array = tf_local.keras.applications.inception_v3.preprocess_input(img_array)
    
    return img_array

@app.route('/predict', methods=['POST'])
def predict():
    if not ensure_model_loaded():
        return jsonify({"error": "Model not loaded properly."}), 500

    if 'image' not in request.files:
        return jsonify({"error": "No image provided."}), 400
        
    file = request.files['image']
    if not file:
        return jsonify({"error": "Empty file."}), 400

    try:
        # Read the image
        img_bytes = file.read()
        image = Image.open(io.BytesIO(img_bytes))
        
        # Preprocess
        processed_image = preprocess_image(image, target_size=(299, 299))
        
        # Predict
        prediction = model.predict(processed_image)
        
        # Interpret result. Assuming binary classification (sigmoid)
        # CIFAKE: usually Real vs Fake.
        # Let's say if it's a 1D output
        if len(prediction[0]) == 1:
            score = float(prediction[0][0])
            # Assuming threshold 0.5: usually 0 is Fake, 1 is Real or vice versa.
            # We'll just return the score and let the frontend format it.
            return jsonify({
                "raw_prediction": score,
                "confidence": max(score, 1 - score),
                "is_ai": score < 0.5 # or score > 0.5 depending on your labels
            })
        else:
            # If it's a categorical output (e.g., softmax with 2 classes)
            predicted_class = int(np.argmax(prediction[0]))
            confidence = float(prediction[0][predicted_class])
            return jsonify({
                "raw_prediction": prediction.tolist(),
                "class": predicted_class,
                "confidence": confidence
            })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
