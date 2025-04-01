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
        for char in char_results[0].boxes.xyxy:
            cx, cy, w, h = map(int, char.tolist())
            cls = int(char_results[0].boxes.cls[0])  # Lớp dự đoán (ký tự)
            detected_chars.append((cx, char_results[0].names[cls]))  # Lưu ký tự với vị trí X

        # Sắp xếp ký tự từ trái sang phải
        detected_chars.sort(key=lambda c: c[0])
        plate_text = "".join(c[1] for c in detected_chars)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        
        text_size = cv2.getTextSize(plate_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
        text_x = x1 + (x2 - x1) // 2 - text_size[0] // 2  # Căn giữa theo chiều ngang
        text_y = y1 + (y2 - y1) // 2 + text_size[1] // 2  # Căn giữa theo chiều dọc
        
        # cv2.rectangle(frame, (text_x - 5, text_y - text_size[1] - 5), (text_x + text_size[0] + 5, text_y + 5), (0, 0, 0), -1)
        # Hiển thị chuỗi ký tự nhận diện trên ảnh
        # cv2.putText(frame, plate_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, plate_text, ((int(plate[0]), int(plate[1]-10))), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)


    # Hiển thị video real-time
    cv2.imshow("License Plate Recognition", frame)

    # Nhấn 'Q' để thoát
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Giải phóng tài nguyên
cap.release()
cv2.destroyAllWindows()
