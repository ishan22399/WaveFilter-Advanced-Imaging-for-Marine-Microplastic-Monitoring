from flask import Flask, request, render_template, jsonify, send_from_directory
import torch
from PIL import Image
import cv2
import numpy as np
import os
import base64
from werkzeug.utils import secure_filename
import pathlib

# Force Windows to use WindowsPath
if os.name == 'nt':
    temp = pathlib.PosixPath
    pathlib.PosixPath = pathlib.WindowsPath

app = Flask(__name__)

# Configure paths and settings
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
RESULTS_FOLDER = os.path.join(BASE_DIR, 'results')
MODEL_PATH = os.path.join(BASE_DIR, 'best_microplastic_model.pt')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Create necessary directories
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Global variable for the model
MODEL = None

def load_model():
    """Load the YOLOv5 model at startup"""
    global MODEL
    try:
        print(f"Loading model from: {MODEL_PATH}")
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model file not found at: {MODEL_PATH}")
            
        MODEL = torch.hub.load('ultralytics/yolov5', 'custom', path=MODEL_PATH)
        print("Model loaded successfully!")
        return True
    except Exception as e:
        print(f"Error loading model: {e}")
        return False

def allowed_file(filename):
    """Check if the file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_image(image_path):
    """Process image and return detection results"""
    try:
        # Load and verify the image
        img_pil = Image.open(image_path).convert("RGB")  # Convert to RGB
        img_cv = cv2.imread(image_path)
        if img_cv is None:
            raise ValueError("OpenCV couldn't load the image")

        # Perform detection
        results = MODEL(img_pil)

        # Draw bounding boxes
        for box in results.xyxy[0]:
            x1, y1, x2, y2, conf, cls = box
            if conf > 0.5:  # Confidence threshold
                label = f'{MODEL.names[int(cls)]} {conf:.2f}'
                cv2.rectangle(img_cv, (int(x1), int(y1)), (int(x2), int(y2)), 
                            (255, 0, 0), 2)
                cv2.putText(img_cv, label, (int(x1), int(y1) - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Save the result image
        result_filename = f'result_{os.path.basename(image_path)}'
        result_path = os.path.join(RESULTS_FOLDER, result_filename)
        cv2.imwrite(result_path, img_cv)

        # Convert result image to base64 for web display
        _, buffer = cv2.imencode('.jpg', img_cv)
        img_base64 = base64.b64encode(buffer).decode('utf-8')

        # Prepare detection results
        results_dict = {
            'detections': [],
            'total_count': len(results.xyxy[0])
        }

        for box in results.xyxy[0]:
            x1, y1, x2, y2, conf, cls = box
            if conf > 0.5:
                results_dict['detections'].append({
                    'class': MODEL.names[int(cls)],
                    'confidence': float(conf),
                    'bbox': [float(x) for x in [x1, y1, x2, y2]]
                })

        return True, results_dict, img_base64, result_path

    except Exception as e:
        return False, str(e), None, None

@app.route('/')
def home():
    """Render the home page"""
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    """Handle image upload and detection"""
    if MODEL is None:
        return jsonify({'error': 'Model not initialized properly'}), 500
    
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No image selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type'}), 400
    
    try:
        # Save uploaded image
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Process the image
        success, results, img_base64, result_path = process_image(filepath)
        
        # Clean up uploaded image
        os.remove(filepath)
        
        if success:
            return jsonify({
                'success': True,
                'results': results,
                'image': img_base64,
                'result_path': result_path
            })
        else:
            return jsonify({'error': results}), 500
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/results')
def results():
    """Render a page displaying all detection result images."""
    try:
        # List all files in the RESULTS_FOLDER
        images = [f for f in os.listdir(RESULTS_FOLDER) if allowed_file(f)]
        return render_template('results.html', images=images)
    except Exception as e:
        return str(e), 500

@app.route('/results/<filename>')
def get_result_image(filename):
    """Serve result images."""
    return send_from_directory(RESULTS_FOLDER, filename)

if __name__ == '__main__':
    print(f"Starting Flask application...")
    print(f"Upload folder: {UPLOAD_FOLDER}")
    print(f"Results folder: {RESULTS_FOLDER}")
    
    # Load model at startup
    if not load_model():
        print("Failed to load model. Please ensure the model file exists and is valid.")
        exit(1)
        
    app.run(debug=True, host='0.0.0.0', port=5000)  # Allow external connections
