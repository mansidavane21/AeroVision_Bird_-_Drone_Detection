📘 AeroVision – Bird & Drone Detection
AeroVision is a complete deep learning project designed for aerial object detection and classification, specifically distinguishing between Birds and Drones using both CNN-based classification models and YOLOv8 object detection.

This project is fully reproducible, well-structured, and ready for GitHub deployment.


🚀 Features
✅ Image Classification Models
Custom CNN
ResNet50
MobileNetV2
EfficientNet-B0

✅ Object Detection
YOLOv8
Bounding-box detection for Birds / Drones
Real-time inference support

✅ Extras
Clean & documented Colab Notebook
Streamlit GUI-ready code
Organized folder structure
Supports custom datasets


📂 Project Structure
AeroVision_Bird_&_Drone_Detection/
│
├── models/                   
│     custom_cnn_best.pth
│     resnet50_best.pth
│     mobilenet_best.pth
│     efficientnet_best.pth
│
├── yolov8/                   
│     yolov8_best.pt
│
├── AeroVision_Bird_&_Drone_Detection.ipynb
├── requirements.txt
├── .gitignore
└── README.md


🧠 Model Training Summary
1️⃣ Classification Models
Notebook includes:
Data preprocessing
Augmentation
Training loops
Accuracy/Loss visualization
Confusion matrix
Model comparison

2️⃣ YOLOv8 Object Detection
Dataset converted to YOLO format
Trained using Ultralytics YOLO
Best model saved at: runs/detect/train/weights/best.pt


📊 Performance Metrics:
| Model           | Accuracy | Loss   | Notes                     |
| --------------- | -------- | ------ | ------------------------- |
| Custom CNN      | 81.00%   | Medium | Lightweight baseline      |
| ResNet50        | 97.96%   | Low    | Best overall accuracy     |
| MobileNetV2     | 95.02%   | Low    | Fast and efficient        |
| EfficientNet-B0 | 94.57%   | Low    | Great balance             |
| YOLOv8          | High mAP | Low    | Best for object detection |


🛠 Installation
Install dependencies:
pip install -r requirements.txt


▶️ YOLO Inference Example
from ultralytics import YOLO
model = YOLO("yolov8/yolov8_best.pt")
results = model("test_image.jpg")
results.show()


🖥 Streamlit App
streamlit run app.py


📦 Dataset
Contains:
Bird images
Drone images
YOLO annotations 
inks:
https://drive.google.com/drive/folders/1nn1vqsh8juhafkJcleembrjQ9EqtIoMh?usp=sharing
https://drive.google.com/drive/folders/114wV_igIhWldcG0HftNIZZsivrs8G22p?usp=sharing


🧩 Technologies Used
Python
PyTorch
Torchvision
Ultralytics YOLO
NumPy / Pandas
Matplotlib
Streamlit
Google Colab


✨ Future Improvements
Drone size estimation
Multi-class detection
Real-time webcam inference
Integration with real surveillance systems


📝 Autho
Mansi
AeroVision — Bird & Drone Detection Project (2025)