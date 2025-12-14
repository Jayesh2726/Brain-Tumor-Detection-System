from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import os
from werkzeug.utils import secure_filename
import tempfile

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configuration
TARGET_SIZE = (64, 64)
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Load your trained model
# UPDATED: Using the fixed model filename
MODEL_PATH = os.environ.get('MODEL_PATH', 'BrainTumor10Epochs.h5')

try:
    # Import after Flask initialization
    from tensorflow import keras
    model = keras.models.load_model(MODEL_PATH, compile=False)
    
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
    """Check if uploaded file has allowed extension"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(img_path):
    """Preprocess the image for model prediction"""
    from tensorflow.keras.preprocessing import image
    
    # Load image
    img = image.load_img(img_path, target_size=TARGET_SIZE, color_mode='rgb')
    
    # Convert to array
    img_array = image.img_to_array(img)
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    # Normalize pixel values to [0, 1]
    img_array = img_array / 255.0
    
    return img_array

@app.route('/')
def home():
    """Home endpoint - API information"""
    return jsonify({
        'message': 'Brain Tumor Detection API',
        'status': 'running',
        'model_loaded': model is not None,
        'model_path': MODEL_PATH,
        'endpoints': {
            'GET /': 'API information',
            'POST /predict': 'Upload MRI image for prediction',
            'GET /health': 'Health check'
        }
    })

@app.route('/predict', methods=['POST'])
def predict():
    """
    Endpoint to predict brain tumor from uploaded MRI image
    
    Request: multipart/form-data with 'image' file
    Response: JSON with prediction results
    """
    
    # Check if model is loaded
    if model is None:
        return jsonify({
            'error': 'Model not loaded. Please check server configuration.',
            'suggestion': 'Make sure BrainTumor10Epochs_fixed.h5 is in the repository'
        }), 500
    
    # Check if image file is present in request
    if 'image' not in request.files:
        return jsonify({
            'error': 'No image file provided',
            'usage': 'Send POST request with form-data containing "image" file'
        }), 400
    
    file = request.files['image']
    
    # Check if file is selected
    if file.filename == '':
        return jsonify({
            'error': 'No file selected'
        }), 400
    
    # Check if file type is allowed
    if not allowed_file(file.filename):
        return jsonify({
            'error': 'Invalid file type. Only PNG, JPG, and JPEG are allowed.',
            'received': file.filename
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
        
        # Determine result based on threshold (0.5)
        has_tumor = prediction_value > 0.5
        
        # Calculate confidence
        if has_tumor:
            confidence = prediction_value * 100
        else:
            confidence = (1 - prediction_value) * 100
        
        # Clean up temporary file
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
        
        # Return results
        return jsonify({
            'success': True,
            'hasTumor': has_tumor,
            'confidence': round(confidence, 2),
            'raw_prediction': round(prediction_value, 4),
            'message': 'Tumor Detected' if has_tumor else 'No Tumor Detected',
            'threshold': 0.5
        })
    
    except Exception as e:
        # Clean up temporary file if it exists
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
        
        return jsonify({
            'error': f'Error processing image: {str(e)}',
            'type': type(e).__name__
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint for monitoring"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'service': 'Brain Tumor Detection API'
    })

if __name__ == '__main__':
    # Get port from environment variable (Render provides this)
    port = int(os.environ.get('PORT', 10000))
    
    # Run the Flask app
    app.run(debug=False, host='0.0.0.0', port=port)