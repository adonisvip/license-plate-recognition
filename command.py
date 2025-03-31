import cv2
import pytesseract
import pytesseractpytesseract
from ultralytics import YOLO
from PIL import Image
import numpy as np
import torch

def detect_and_recognize_plate(image_path, output_path="detected_result.png", model_path="model/LP_detector.pt"):
    model = YOLO(model_path)
    
    image = cv2.imread(image_path)
    if image is None:
        print("Lỗi: Không thể đọc ảnh!")
        return
    
    results = model(image)
    
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Lấy tọa độ biển số
            cropped_plate = image[y1:y2, x1:x2]  # Cắt vùng biển số
            
            cropped_plate_gray = cv2.cvtColor(cropped_plate, cv2.COLOR_BGR2GRAY)
            _, cropped_plate_thresh = cv2.threshold(cropped_plate_gray, 150, 255, cv2.THRESH_BINARY)
            
            plate_number = pytesseract.image_to_string(cropped_plate_thresh, config="--psm 7 --oem 3")
            plate_number = plate_number.strip()
            
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            cv2.putText(image, plate_number, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
            print(f"📸 Biển số nhận diện: {plate_number}")
    
    cv2.imwrite(output_path, image)
    print(f"✅ Ảnh kết quả đã lưu tại: {output_path}")
    
    cv2.imshow("License Plate Detection", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect and recognize license plates from an image.")
    parser.add_argument("--image", type=str, required=True, help="Path to the input image.")
    parser.add_argument("--output", type=str, default="detected_result.png", help="Path to save the output image.")
    args = parser.parse_args()
    
    detect_and_recognize_plate(args.image, args.output)
