from flask import Flask, request, jsonify
import os
from PIL import Image
import torch
from ultralytics import YOLO

# Initialize the Flask application
app = Flask(__name__)

# Load the YOLO model
model = YOLO('models/yolo11n.pt')  # Specify the path to your YOLO model weights

@app.route('/')
def home():
    return '''
        <h1>Object Detection App</h1>
        <p>Upload an image to detect objects.</p>
        <form action="/predict" method="post" enctype="multipart/form-data">
            <input type="file" name="file" />
            <input type="submit" value="Upload Image" />
        </form>
    '''

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    # Open the image file
    try:
        image = Image.open(file.stream)
    except Exception as e:
        return jsonify({'error': str(e)}), 400

    # Perform inference
    results = model.predict(image)

    # Get the predictions
    predictions = []
    for result in results:
        for box in result.boxes:
            predictions.append({
                'class': box.cls.item(),
                'confidence': box.conf.item(),
                'xyxy': box.xyxy.tolist()  # Convert to list for JSON serialization
            })

    return jsonify(predictions)

if __name__ == '__main__':
    # Set the host and port
    app.run(host='0.0.0.0', port=5000, debug=True)
