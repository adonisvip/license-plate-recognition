import cv2
from ultralytics import YOLO
from function.sort_charater import sort_by_rows
from function.process_frame import process_frame

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

    processed_frame = process_frame(frame, plate_detector, char_detector)
    cv2.imshow("License Plate Recognition", processed_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
