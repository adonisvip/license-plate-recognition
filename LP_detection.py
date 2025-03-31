import cv2
import argparse
import numpy as np
from ultralytics import YOLO
import function.helper as helper
import function.utils_rotate as utils_rotate
# Load YOLO models

plate_detector = YOLO("model/LP_detector.pt")  # Model phát hiện biển số
character_recognizer = YOLO("model/LP_ocr.pt")  # Model nhận diện ký tự trên biển số
character_recognizer.conf = 0.60

def detect_and_recognize_plate(image_path, output_path="detected_result.png"):
    image = cv2.imread(image_path)
    if image is None:
        print("Lỗi: Không thể đọc ảnh!")
        return
    
    results = plate_detector(image)
    plates_detected = []
    
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Lấy tọa độ biển số
            cropped_plate = image[y1:y2, x1:x2]  # Cắt vùng biển số
            
            # Nhận diện ký tự bằng YOLO
            char_results = character_recognizer(cropped_plate)
            plate_number = ""
            
            for char_r in char_results:
                for char_box in char_r.boxes:
                    char_x1, char_y1, char_x2, char_y2 = map(int, char_box.xyxy[0])
                    char_crop = cropped_plate[char_y1:char_y2, char_x1:char_x2]
                    plate_number += char_r.names[int(char_box.cls[0])]  # Lấy tên lớp dự đoán
            
            # Vẽ biển số và hiển thị số trên ảnh
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, plate_number, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            plates_detected.append(plate_number)
    
    cv2.imwrite(output_path, image)
    print(f"📸 Biển số xe phát hiện: {plates_detected}")
    print(f"✅ Ảnh kết quả đã lưu tại: {output_path}")
    
    cv2.imshow("License Plate Recognition", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect and recognize license plates from an image using YOLO.")
    parser.add_argument("--image", type=str, required=True, help="Path to the input image.")
    parser.add_argument("--output", type=str, default="detected_result.png", help="Path to save the output image.")
    args = parser.parse_args()
    
    detect_and_recognize_plate(args.image, args.output)