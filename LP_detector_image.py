import argparse
import cv2
import torch
from ultralytics import YOLO
from function.sort_charater import sort_by_rows

parser = argparse.ArgumentParser(description="License Plate Recognition with YOLOv8")
parser.add_argument("--image", type=str, required=True, help="Path to input image")
parser.add_argument("--output", type=str, default="detected_result.png", help="Path to save output image")
args = parser.parse_args()

plate_detector = YOLO("model/LP_detector.pt") 
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
    char_results = char_detector(plate_img)
    detected_chars = []

    for box, conf, cls in zip(char_results[0].boxes.xyxy, char_results[0].boxes.conf, char_results[0].boxes.cls):
        x_min, y_min, x_max, y_max = map(int, box.tolist())
        label = char_results[0].names[int(cls)]

        if conf > 0.5:  # Lọc ký tự có độ tin cậy cao
            detected_chars.append(((x_min, y_min, y_max), label))  # Lưu (x_min, y_min, y_max)

    # ======== SẮP XẾP KÝ TỰ ========

    # Nhận diện biển số đầy đủ
    plate_text = sort_by_rows(detected_chars)
    print("Plate text:", plate_text)

    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv2.putText(image, plate_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
    
# Lưu ảnh kết quả
cv2.imwrite(args.output, image)
print(f"✅ Ảnh kết quả đã được lưu tại {args.output}")
cv2.imshow('frame', image)
cv2.waitKey()
cv2.destroyAllWindows()