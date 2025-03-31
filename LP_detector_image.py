import argparse
import cv2
import torch
import numpy as np
from ultralytics import YOLO

parser = argparse.ArgumentParser(description="License Plate Recognition with YOLOv8")
parser.add_argument("--image", type=str, required=True, help="Path to input image")
parser.add_argument("--output", type=str, default="detected_result.png", help="Path to save output image")
args = parser.parse_args()

plate_detector = YOLO("model/LP_detector.pt")  # Mô hình nhận diện biển số
char_detector = YOLO("model/LP_ocr.pt")  

image = cv2.imread(args.image)
if image is None:
    raise ValueError("Không thể đọc ảnh. Kiểm tra đường dẫn!")

# ======== BƯỚC 1: Phát hiện biển số ========
plate_results = plate_detector(image)
for plate in plate_results[0].boxes.xyxy:
    x1, y1, x2, y2 = map(int, plate.tolist())
    
    # Cắt ảnh biển số
    plate_img = image[y1:y2, x1:x2].copy()
    cv2.imwrite("crop.jpg", plate_img)
    if plate_img is None or plate_img.size == 0:
        print("⚠ Không thể cắt biển số, bỏ qua.")
        continue

    # ======== BƯỚC 2: Phát hiện ký tự trên biển số ========
    # rc_image = cv2.imread("crop.jpg")
    char_results = char_detector(plate_img)

    detected_chars = []
    for char in char_results[0].boxes.xyxy:
        cx, cy, w, h = map(int, char.tolist())
        cls = int(char_results[0].boxes.cls[0])
        detected_chars.append((cx, char_results[0].names[cls])) 

    detected_chars.sort(key=lambda c: c[0])
    plate_text = "".join(c[1] for c in detected_chars)
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
    
    cv2.putText(image, plate_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

# Lưu ảnh kết quả
cv2.imwrite(args.output, image)
print(f"✅ Ảnh kết quả đã được lưu tại {args.output}")
cv2.imshow('frame', image)
cv2.waitKey()
cv2.destroyAllWindows()