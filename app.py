from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import os
from werkzeug.utils import secure_filename
import tempfile

app = Flask(__name__)
CORS(app)

# ---------------- CONFIG ----------------
TARGET_SIZE = (64, 64)
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB

BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, 'BrainTumor10Epochs_fixed.h5')

# --------- LAZY MODEL LOADING ----------
model = None

def get_model():
    global model
    if model is None:
        print("🔥 Loading model...")
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        print("✅ Model loaded successfully")
    return model

# ---------------------------------------

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=TARGET_SIZE, color_mode='rgb')
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

# ---------------- ROUTES ----------------

@app.route('/')
def home():
    return jsonify({
        'message': 'Brain Tumor Detection API',
        'status': 'running'
    })

@app.route('/health')
def health():
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None
    })

@app.route('/predict', methods=['POST'])
def predict():
    try:
        model_instance = get_model()

        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400

        file = request.files['image']

        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type'}), 400

        filename = secure_filename(file.filename)
        temp_path = os.path.join(tempfile.gettempdir(), filename)
        file.save(temp_path)

        img = preprocess_image(temp_path)
        prediction = model_instance.predict(img, verbose=0)[0][0]

        os.remove(temp_path)

        has_tumor = prediction > 0.5
        confidence = prediction * 100 if has_tumor else (1 - prediction) * 100

        return jsonify({
            'success': True,
            'hasTumor': has_tumor,
            'confidence': round(confidence, 2),
            'raw_prediction': round(float(prediction), 4),
            'message': 'Tumor Detected' if has_tumor else 'No Tumor Detected'
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ---------------------------------------

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)
