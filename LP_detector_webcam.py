import cv2
import torch
from ultralytics import YOLO

plate_detector = YOLO("model/LP_detector.pt")  # Mô hình nhận diện biển số
char_detector = YOLO("model/LP_ocr.pt")  

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Không thể mở camera!")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Không thể lấy khung hình từ camera!")
        break

    # ======== BƯỚC 1: Phát hiện biển số ========
    plate_results = plate_detector(frame)
    for plate in plate_results[0].boxes.xyxy:
        x1, y1, x2, y2 = map(int, plate.tolist())
        
        # Cắt ảnh biển số
        plate_img = frame[y1:y2, x1:x2].copy()
        cv2.imwrite("crop.jpg", plate_img)

        # ======== BƯỚC 2: Phát hiện ký tự trên biển số ========
        char_results = char_detector(plate_img)

        detected_chars = []
        for char, conf, cls in zip(char_results[0].boxes.xyxy, char_results[0].boxes.conf, char_results[0].boxes.cls):
            cx, cy, w, h = map(int, char.tolist())
            label = char_results[0].names[int(cls)]

            if conf > 0.5:  # Lọc ký tự có độ tin cậy cao
                detected_chars.append((cx, label, conf))
                cv2.rectangle(plate_img, (cx, cy), (w, h), (0, 255, 0), 2)
                cv2.putText(plate_img, label, (cx, cy - 5), cv2.FONT_HERSHEY_SIMPLEX, 
                            0.8, (0, 255, 0), 2)
        # Sắp xếp ký tự từ trái sang phải
        detected_chars.sort(key=lambda c: c[0])
        #plate_text = "".join(c[1] for c in detected_chars)
            
        plate_text = "".join(c[1] for c in detected_chars)
        cv2.putText(frame, plate_text, ((int(plate[0]), int(plate[1]-10))), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

    # Hiển thị video real-time
    cv2.imshow("License Plate Recognition", frame)

    # Nhấn 'Q' để thoát
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Giải phóng tài nguyên
cap.release()
cv2.destroyAllWindows()
