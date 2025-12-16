from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from keras.preprocessing import image
from keras.models import load_model
import os
from werkzeug.utils import secure_filename
import tempfile

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configuration
TARGET_SIZE = (64, 64)
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

# Model path - use the FIXED model file
MODEL_PATH = os.environ.get('MODEL_PATH', 'BrainTumor10Epochs_fixed.h5')

# Load model
model = None
try:
    model = load_model(MODEL_PATH, compile=False)
    
    # Recompile the model
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    print("✅ Model loaded successfully!")
    print(f"Model path: {MODEL_PATH}")
    
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(img_path):
    """Preprocess the image for model prediction"""
    img = image.load_img(img_path, target_size=TARGET_SIZE, color_mode='rgb')
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize pixel values to [0, 1]
    return img_array

@app.route('/')
def home():
    """API home endpoint"""
    return jsonify({
        'message': 'Brain Tumor Detection API',
        'status': 'running',
        'model_loaded': model is not None,
        'model_path': MODEL_PATH
    })

@app.route('/predict', methods=['POST'])
def predict():
    """
    Endpoint to predict brain tumor from uploaded MRI image
    Expects: multipart/form-data with 'image' file
    Returns: JSON with prediction results
    """
    
    # Check if model is loaded
    if model is None:
        return jsonify({
            'error': 'Model not loaded. Please check server configuration.'
        }), 500
    
    # Check if image file is present in request
    if 'image' not in request.files:
        return jsonify({
            'error': 'No image file provided'
        }), 400
    
    file = request.files['image']
    
    # Check if file is selected
    if file.filename == '':
        return jsonify({
            'error': 'No file selected'
        }), 400
    
    # Check if file is allowed
    if not allowed_file(file.filename):
        return jsonify({
            'error': 'Invalid file type. Only PNG, JPG, and JPEG are allowed.'
        }), 400
    
    temp_path = None
    try:
        # Save file temporarily
        filename = secure_filename(file.filename)
        temp_dir = tempfile.gettempdir()
        temp_path = os.path.join(temp_dir, filename)
        file.save(temp_path)
        
        # Preprocess the image
        processed_image = preprocess_image(temp_path)
        
        # Make prediction
        prediction = model.predict(processed_image, verbose=0)
        prediction_value = float(prediction[0][0])
        
        # Determine result based on threshold
        has_tumor = prediction_value > 0.5
        confidence = prediction_value * 100 if has_tumor else (1 - prediction_value) * 100
        
        # Clean up temporary file
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
        
        # Return results
        return jsonify({
            'success': True,
            'hasTumor': has_tumor,
            'confidence': round(confidence, 2),
            'raw_prediction': round(prediction_value, 4),
            'message': 'Tumor Detected' if has_tumor else 'No Tumor Detected'
        })
    
    except Exception as e:
        # Clean up temporary file if it exists
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
        
        return jsonify({
            'error': f'Error processing image: {str(e)}'
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None
    })

if __name__ == '__main__':
    # Get port from environment variable (Render provides this)
    port = int(os.environ.get('PORT', 10000))
    app.run(debug=False, host='0.0.0.0', port=port)