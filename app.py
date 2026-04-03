import os
import io
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
from PIL import Image


def load_model_with_compatibility(model_path: str):
    """Load model with compatibility fallbacks for common Keras serialization mismatches."""
    custom_objects = {
        # Newer saved configs may include this class name while older runtimes expect Policy.
        "DTypePolicy": tf.keras.mixed_precision.Policy,
    }

    attempts = []

    def _attempt_default():
        return tf.keras.models.load_model(model_path, compile=False)

    def _attempt_custom_objects():
        return tf.keras.models.load_model(
            model_path,
            compile=False,
            custom_objects=custom_objects,
        )

    def _attempt_inputlayer_patch(use_custom_objects: bool):
        original_init = tf.keras.layers.InputLayer.__init__

        def patched_init(self, *args, **kwargs):
            if "batch_shape" in kwargs and "batch_input_shape" not in kwargs:
                kwargs["batch_input_shape"] = kwargs.pop("batch_shape")
            return original_init(self, *args, **kwargs)

        tf.keras.layers.InputLayer.__init__ = patched_init
        try:
            if use_custom_objects:
                return tf.keras.models.load_model(
                    model_path,
                    compile=False,
                    custom_objects=custom_objects,
                )
            return tf.keras.models.load_model(model_path, compile=False)
        finally:
            tf.keras.layers.InputLayer.__init__ = original_init

    loaders = [
        _attempt_default,
        _attempt_custom_objects,
        lambda: _attempt_inputlayer_patch(use_custom_objects=False),
        lambda: _attempt_inputlayer_patch(use_custom_objects=True),
    ]

    for loader in loaders:
        try:
            return loader()
        except Exception as err:
            attempts.append(str(err))

    raise RuntimeError(" | ".join(attempts))

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


@app.route('/health', methods=['GET'])
def health():
    loaded = ensure_model_loaded()
    status_code = 200 if loaded else 500
    return jsonify({
        "status": "ok" if loaded else "error",
        "model_loaded": loaded,
        "model_load_error": model_load_error
    }), status_code

def preprocess_image(image, target_size=(299, 299)):
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(target_size)
    img_array = np.array(image)
    
    # Expand dimensions to create a batch size of 1
    img_array = np.expand_dims(img_array, axis=0)
    
    # InceptionV3 preprocessing
    # Or simply: img_array = img_array / 255.0 depending on how the model was trained.
    img_array = tf.keras.applications.inception_v3.preprocess_input(img_array)
    
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
