import cv2
import torch
from ultralytics import YOLO
from function.sort_charater import sort_by_rows

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
        for box, conf, cls in zip(char_results[0].boxes.xyxy, char_results[0].boxes.conf, char_results[0].boxes.cls):
            x_min, y_min, x_max, y_max = map(int, box.tolist())
            label = char_results[0].names[int(cls)]

            if conf > 0.5:  # Lọc ký tự có độ tin cậy cao
                detected_chars.append(((x_min, y_min, y_max), label))

        plate_text = sort_by_rows(detected_chars)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(frame, plate_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

    cv2.imshow("License Plate Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
