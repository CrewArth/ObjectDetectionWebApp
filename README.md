##Object Detection Web App
This project is an Object Detection Web App built using a fine-tuned YOLO (You Only Look Once) model specifically trained on the BCCD (Blood Cell Count and Detection) dataset. The app allows users to upload images and detect various types of blood cells, showcasing the power of deep learning and computer vision in biomedical applications.

Features
Upload and Detect: Users can upload images of blood samples, and the app will detect and highlight different types of blood cells in real-time.
Model Training: The YOLO model has been fine-tuned on the BCCD dataset, improving accuracy in identifying various cell types.
User-Friendly Interface: The web app provides a simple and intuitive interface for users to interact with the model and view results.
Technologies Used
YOLOv5: A state-of-the-art object detection model that is fast and accurate.
Flask: A lightweight WSGI web application framework for Python.
OpenCV: A library for computer vision tasks to process images.
HTML/CSS: For building the front-end interface.
JavaScript: For enhancing user experience on the web app.
Setup
Clone this repository and navigate into the project directory.

bash
Copy code
git clone <repo-url>
cd object_detection_webapp
Install the required dependencies:

bash
Copy code
pip install -r requirements.txt
Run the application:

bash
Copy code
python app.py
Open your web browser and navigate to http://127.0.0.1:5000 to access the app.

Usage
Upload an image containing blood cells.
Click on the "Detect" button to start the detection process.
View the results with detected cells highlighted on the image.
Contributing
Contributions are welcome! Please feel free to submit a pull request or open an issue for any enhancements or bugs.

License
This project is licensed under the MIT License. See the LICENSE file for details.
