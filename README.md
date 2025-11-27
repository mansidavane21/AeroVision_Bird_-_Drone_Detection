<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Deep%20Learning-PyTorch-red?style=for-the-badge&logo=pytorch" />
  <img src="https://img.shields.io/badge/Object%20Detection-YOLOv8-yellow?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Status-Active-success?style=for-the-badge" />
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" />
</p>

# 📘 AeroVision – Bird & Drone Detection

**AeroVision** is a complete deep learning project designed for **aerial object detection and classification**, specifically distinguishing between **Birds and Drones**. It leverages both **CNN-based classification models** and **YOLOv8 object detection** to deliver high-accuracy predictions.

The project is fully reproducible, well-structured, and ready for GitHub deployment.

---

# 🚀 Features

### **1️⃣ Image Classification**

* Custom CNN
* ResNet50
* MobileNetV2
* EfficientNet-B0

### **2️⃣ Object Detection**

* YOLOv8
* Bounding-box detection for Birds / Drones
* Real-time inference support

### **3️⃣ Extras**

* Clean & well-documented Colab Notebook
* Streamlit GUI-ready code
* Organized folder structure
* Supports custom datasets

---

# 📂 Project Structure

```
AeroVision_Bird_&_Drone_Detection/
│
├── models/                   
│     ├── custom_cnn_best.pth
│     ├── resnet50_best.pth
│     ├── mobilenet_best.pth
│     └── efficientnet_best.pth
│
├── yolov8/                   
│     └── yolov8_best.pt
│
├── AeroVision_Bird_&_Drone_Detection.ipynb
├── requirements.txt
├── .gitignore
└── README.md
```

---

# 🧠 Model Training Summary

### **1️⃣ Classification Models**

* Data preprocessing & augmentation
* Training loops with accuracy/loss visualization
* Confusion matrix
* Model comparison

### **2️⃣ YOLOv8 Object Detection**

* Dataset converted to YOLO format
* Trained using **Ultralytics YOLOv8**
* Best model saved at: `runs/detect/train/weights/best.pt`

---

# 📊 Performance Metrics

| Model           | Accuracy | Loss   | Notes                     |
| --------------- | -------- | ------ | ------------------------- |
| Custom CNN      | 81.00%   | Medium | Lightweight baseline      |
| ResNet50        | 97.96%   | Low    | Best overall accuracy     |
| MobileNetV2     | 95.02%   | Low    | Fast and efficient        |
| EfficientNet-B0 | 94.57%   | Low    | Great balance             |
| YOLOv8          | High mAP | Low    | Best for object detection |

---

# 🛠 Installation

### **1️⃣ Clone Repository**

```bash
git clone https://github.com/mansidavane21/AeroVision_Bird_Drone_Detection.git
cd AeroVision_Bird_&_Drone_Detection
```

### **2️⃣ Install Dependencies**

```bash
pip install -r requirements.txt
```

---

# ▶️ YOLO Inference Example

```python
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO("yolov8/yolov8_best.pt")

# Perform inference on a test image
results = model("test_image.jpg")
results.show()
```

---

# 🖥 Streamlit App

```bash
streamlit run app.py
```

*Provides a user-friendly interface for single-image or batch predictions.*

---

# 📦 Dataset

Includes:

* Bird images
* Drone images
* YOLO annotations

**Links:**

* [Classification Dataset](https://drive.google.com/drive/folders/1nn1vqsh8juhafkJcleembrjQ9EqtIoMh?usp=sharing)
* [YOLO Object Detection Dataset](https://drive.google.com/drive/folders/114wV_igIhWldcG0HftNIZZsivrs8G22p?usp=sharing)

---

# 🧩 Technologies Used

* Python
* PyTorch & Torchvision
* Ultralytics YOLOv8
* NumPy / Pandas
* Matplotlib / Seaborn
* Streamlit
* Google Colab

---

# ✨ Future Improvements

* Drone size estimation
* Multi-class detection (e.g., different bird species)
* Real-time webcam inference
* Integration with real surveillance systems

---

# 🤝 Contributing

We welcome contributions!

**Steps:**

1. Fork the project
2. Create a feature branch

```bash
git checkout -b feature-name
```

3. Commit changes

```bash
git commit -m "Add new feature"
```

4. Push & create a Pull Request

Follow **PEP8 coding standards** for all contributions.

---

# 👤 Author

**Mansi Davane** — Data Science student & developer (B.Tech, Data Science)
This project demonstrates **real-world application of deep learning for aerial object detection**.

---
