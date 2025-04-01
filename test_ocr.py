import cv2
from ultralytics import YOLO

# Load mô hình OCR
char_detector = YOLO("model/last.pt")  

# Đọc ảnh biển số (ảnh đã cắt)
plate_img = cv2.imread("crop.jpg")  # Thay bằng đường dẫn ảnh biển số
if plate_img is None:
    raise ValueError("❌ Không thể đọc ảnh biển số! Kiểm tra đường dẫn!")

# Chạy mô hình nhận diện ký tự
char_results = char_detector(plate_img)

# Hiển thị kết quả nhận diện
detected_chars = []
for char, conf, cls in zip(char_results[0].boxes.xyxy, char_results[0].boxes.conf, char_results[0].boxes.cls):
    cx, cy, w, h = map(int, char.tolist())
    label = char_results[0].names[int(cls)]

    if conf > 0.5:  # Lọc ký tự có độ tin cậy cao
        detected_chars.append((cx, label, conf))
        cv2.rectangle(plate_img, (cx, cy), (w, h), (0, 255, 0), 2)
        cv2.putText(plate_img, label, (cx, cy - 5), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.8, (0, 255, 0), 2)

# Sắp xếp theo vị trí để ghép lại thành biển số hoàn chỉnh
detected_chars.sort(key=lambda c: c[0]) 
plate_text = "".join(c[1] for c in detected_chars)

print(f"📌 Biển số OCR nhận diện: {plate_text}")

# Hiển thị ảnh
cv2.imshow("OCR Test", plate_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
