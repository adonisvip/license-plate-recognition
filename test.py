import argparse
import cv2
import torch
import numpy as np
from ultralytics import YOLO

parser = argparse.ArgumentParser(description="License Plate Recognition with YOLOv8")
parser.add_argument("--image", type=str, required=True, help="Path to input image")
parser.add_argument("--output", type=str, default="detected_result.png", help="Path to save output image")
args = parser.parse_args()

# Load YOLO models
plate_detector = YOLO("model/LP_detector.pt")  
char_detector = YOLO("model/LP_ocr.pt")  

# Read input image
image = cv2.imread(args.image)
if image is None:
    raise ValueError("❌ Không thể đọc ảnh. Kiểm tra đường dẫn!")

# ======== BƯỚC 1: Phát hiện biển số ========
plate_results = plate_detector(image)
for plate in plate_results[0].boxes.xyxy:
    x1, y1, x2, y2 = map(int, plate.tolist())
    
    # Cắt ảnh biển số
    plate_img = image[y1:y2, x1:x2].copy()
    if plate_img is None or plate_img.size == 0:
        print("⚠ Không thể cắt biển số, bỏ qua.")
        continue

    # ======== BƯỚC 2: Nhận diện ký tự trên biển số ========
    char_results = char_detector(plate_img)
    detected_chars = []

    for char, conf, cls in zip(char_results[0].boxes.xyxy, char_results[0].boxes.conf, char_results[0].boxes.cls):
        cx, cy, w, h = map(int, char.tolist())
        label = char_results[0].names[int(cls)]

        if conf > 0.5:  # Lọc các ký tự có độ tin cậy cao
            detected_chars.append((cx, label, conf))

    # ======== BƯỚC 3: Lọc ký tự bị lặp & nhiễu ========
    detected_chars.sort(key=lambda c: c[0])  # Sắp xếp theo vị trí từ trái qua phải

    filtered_chars = []
    last_x = -999  
    seen_chars = set()

    for cx, char, conf in detected_chars:
        if abs(cx - last_x) > 5 and char not in seen_chars:  
            filtered_chars.append(char)
            seen_chars.add(char)
            last_x = cx  

    plate_text = "".join(filtered_chars)  
    print("📌 Biển số nhận diện:", plate_text)

    # ======== BƯỚC 4: Vẽ lên ảnh ========
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Vẽ khung biển số

    text_size = cv2.getTextSize(plate_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
    text_x = x1 + (x2 - x1) // 2 - text_size[0] // 2  
    text_y = y2 + 30  

    # Vẽ nền đen để hiển thị rõ chữ
    cv2.rectangle(image, (text_x - 5, text_y - text_size[1] - 5), 
                  (text_x + text_size[0] + 5, text_y + 5), (0, 0, 0), -1)

    cv2.putText(image, plate_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 
                1, (0, 255, 0), 2)

# Lưu và hiển thị ảnh kết quả
cv2.imwrite(args.output, image)
print(f"✅ Ảnh kết quả đã được lưu tại {args.output}")
cv2.imshow('License Plate Recognition', image)
cv2.waitKey()
cv2.destroyAllWindows()
