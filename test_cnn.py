import cv2
import numpy as np
import torch
import argparse
from ultralytics import YOLO
from tensorflow.keras.models import load_model

# Load YOLO model for license plate detection
yolo_model = YOLO("model/LP_detector.pt")  # Ensure you have the trained model

# Load CNN model for character recognition
cnn_model = load_model("model/LP_ocr.h5")  # Ensure this is a trained model

def extract_license_plate(image):
    """Detect license plate using YOLO and return cropped plate image."""
    results = yolo_model(image)
    for result in results:
        for box in result.boxes.xyxy:  # Extract bounding boxes
            x1, y1, x2, y2 = map(int, box)
            plate_img = image[y1:y2, x1:x2]
            return plate_img
    return None

def segment_characters(plate_image):
    """Extract characters from license plate using OpenCV."""
    gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    char_images = [cv2.boundingRect(c) for c in contours]
    char_images = sorted(char_images, key=lambda x: x[0])  # Sort from left to right
    
    characters = [plate_image[y:y+h, x:x+w] for (x, y, w, h) in char_images]
    return characters

def recognize_character(char_img):
    """Predict a character from CNN model."""
    char_img = cv2.resize(char_img, (28, 28))  # Resize về đúng kích thước yêu cầu của CNN
    char_img = cv2.cvtColor(char_img, cv2.COLOR_BGR2RGB)  # Chuyển về RGB (3 kênh)
    char_img = char_img.astype("float32") / 255.0  # Chuẩn hóa ảnh về khoảng [0,1]
    char_img = np.expand_dims(char_img, axis=0)  # Định dạng (1, 28, 28, 3)

    # Dự đoán
    features = cnn_model.predict(char_img)  

    # Kiểm tra nếu model không có Flatten, cần reshape thủ công
    if features.shape[1] != 4608:  
        features = features.flatten().reshape(1, -1)  # Reshape thành (1, 4608)

    return chr(np.argmax(features) + ord('0'))  # Convert index to character

def process_image(image_path, output_path):
    """Main pipeline: Detect plate -> Extract characters -> Recognize text."""
    image = cv2.imread(image_path)
    plate_img = extract_license_plate(image)
    if plate_img is None:
        print("No license plate detected!")
        return
    
    chars = segment_characters(plate_img)
    plate_text = "".join([recognize_character(char) for char in chars])
    
    # Draw detected text on image
    cv2.putText(image, plate_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Save or display result
    if output_path:
        cv2.imwrite(output_path, image)
        print(f"Output saved to {output_path}")
    else:
        cv2.imshow("License Plate Recognition", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="car.jpg", help="Path to input image")
    parser.add_argument("--output", type=str, default=None, help="Path to save output image")
    args = parser.parse_args()
    
    process_image(args.input, args.output)