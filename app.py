import os
import io
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
from PIL import Image

app = Flask(__name__)
# Enable CORS for the React frontend
CORS(app)

# Load the model
# Using a try-except block just in case
try:
    model = tf.keras.models.load_model('final_inception_model.h5')
    print("Model loaded successfully!")
except Exception as e:
    print("Error loading model:", e)
    model = None

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
    if model is None:
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
    app.run(debug=True, port=5000)
