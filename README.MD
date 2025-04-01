# License Plate Recognition with YOLOv8

This project implements an automatic **License Plate Recognition (LPR)** system using **YOLOv8** for both plate detection and character recognition.

## 🚀 Features
- Detects vehicle license plates using YOLOv8.
- Recognizes characters on the plate using a trained character recognition model.
- Works with images and real-time video streams.

---
## 📦 Installation
### **1. Install Python & Dependencies**
Ensure you have **Python 3.0+** installed.

```bash
# Install required packages
pip install -r requirement.txt
```

### **2. Download Datasets**
You need two datasets for training:

1. **License Plate Detection Dataset**
   - [Download Here](https://drive.google.com/file/d/106uslUUoWYNJys5JUOMV6-GbuB65fSdt/view?usp=sharing)
   - Extract and place in `datasets/license_plate_detection/`

2. **Character Recognition Dataset**
   - [Download Here](https://drive.google.com/file/d/1IwWqcjGAAuNvY3Dfi5oR5RhiIFaRWG8q/view?usp=sharing)
   - Extract and place in `datasets/character_recognition/`

---
## 🔥 Training YOLOv8 Model
### **1. Train License Plate Detection Model**
```bash
# Train YOLOv8 for license plate detection
yolo train model=yolov8n.pt data=datasets/license_plate_detection/data.yaml epochs=30 imgsz=320 batch=16
```

### **2. Train Character Recognition Model**
```bash
# Train YOLOv8 for character recognition
yolo train model=yolov8n.pt data=datasets/character_recognition/data.yaml epochs=30 imgsz=640 batch=16
```

---
## 🏎 Running the Application
### **1. Detect License Plates in an Image**
```bash
python LP_detector_image.py --image path/to/your/image.jpg --output result.jpg
```

### **3. Real-Time Detection (Webcam)**
```bash
python LP_detector_webcam.py
```

---
## 📊 Results & Evaluation
- Model performance is evaluated using mAP (Mean Average Precision), Precision, and Recall.
- Fine-tuning is recommended if accuracy is low in real-world scenarios.
- Test:
![alt text](image.png)
![alt text](image-1.png)
---
## 🤝 Contributing
Feel free to contribute improvements or report issues!
