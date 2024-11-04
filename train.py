# train.py
import os
import glob
import xml.etree.ElementTree as ET
import torch
from torch.utils.data import Dataset, DataLoader
from ultralytics import YOLO
from albumentations import Compose, Rotate, RandomCrop, HorizontalFlip
from albumentations.pytorch import ToTensorV2
import cv2

# Define transformations
transform = Compose([
    Rotate(limit=20, p=0.5),
    RandomCrop(width=450, height=450, p=0.5),
    HorizontalFlip(p=0.5),
    ToTensorV2()
])

# Dataset class for BCCD Dataset
class BCCDDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_paths = glob.glob(os.path.join(data_dir, 'images', '*.jpg'))
        self.label_paths = [self.get_label_path(p) for p in self.image_paths]

        print(f"Found {len(self.image_paths)} images.")
        print(f"Label paths: {self.label_paths}")

    def get_label_path(self, image_path):
        label_path = os.path.join(
            self.data_dir,
            'annotations',
            os.path.basename(image_path).replace('.jpg', '.xml')
        )
        if not os.path.exists(label_path):
            print(f"Warning: Label file does not exist for {image_path}")
        return label_path

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = cv2.imread(self.image_paths[idx])
        if image is None:
            print(f"Error loading image {self.image_paths[idx]}")
            return None  # Skip this sample

        label_path = self.label_paths[idx]
        boxes, classes = self.load_labels(label_path)

        if not boxes:  # If no boxes, skip this sample
            return None  # Skip this sample

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']

        return image, {'boxes': torch.tensor(boxes, dtype=torch.float32), 'labels': torch.tensor(classes, dtype=torch.long)}

    def load_labels(self, label_path):
        boxes = []
        classes = []

        # Parse XML file
        tree = ET.parse(label_path)
        root = tree.getroot()

        for obj in root.findall('object'):
            # Extract class label
            cls = obj.find('name').text
            classes.append(self.class_to_index(cls))

            # Extract bounding box coordinates
            bbox = obj.find('bndbox')
            x_min = float(bbox.find('xmin').text)
            y_min = float(bbox.find('ymin').text)
            x_max = float(bbox.find('xmax').text)
            y_max = float(bbox.find('ymax').text)
            boxes.append([x_min, y_min, x_max, y_max])

        return boxes, classes

    def class_to_index(self, class_name):
        class_map = {"RBC": 0, "WBC": 1, "Platelets": 2}  # Update based on your classes
        return class_map[class_name]

# Training function
def train(data_dir, epochs=10, batch_size=8):
    # Load dataset and apply transforms
    dataset = BCCDDataset(data_dir, transform=transform)

    # Filter out None samples
    dataset = [sample for sample in dataset if sample is not None]
    
    # Convert the filtered dataset to a DataLoader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Load the YOLO model
    model = YOLO('yolo11n.pt')  # Load pre-trained weights or define your model

    model.train(data='bcdd_dataset.yaml', epochs=10)

    for epoch in range(epochs):
        for batch in dataloader:
            images, targets = batch
            loss = model(images, targets)  # Calculate loss
            loss.backward()                 # Backpropagation
            model.optimizer.step()          # Update weights
            model.optimizer.zero_grad()     # Reset gradients
            print(f"Epoch {epoch}, Loss: {loss.item()}")

    # Create models directory if it doesn't exist
    if not os.path.exists('models'):
        os.makedirs('models')

    # Save the fine-tuned model
    model.save('models/yolov10_bccd.pt')
    print("Model saved successfully at models/yolo11n_bccd.pt")

# Run training
if __name__ == "__main__":
    train('data/BCCD_Dataset', epochs=10)
