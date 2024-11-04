# inference.py
import torch
from ultralytics import YOLO
import cv2

# Load the fine-tuned model
model = YOLO('models/yolov10_bccd.pth')

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    # Apply any required preprocessing here
    return image

def predict(image):
    results = model(image)
    output = []
    for res in results:
        output.append({
            'class': res['class'],
            'confidence': res['confidence'],
            'bbox': res['bbox']
        })
    return output

# Test the inference
if __name__ == "__main__":
    image_path = 'path/to/test_image.jpg'
    image = preprocess_image(image_path)
    results = predict(image)
    print(results)

model.save("models/yolov10_bccd.pt")

