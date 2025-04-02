import argparse
import cv2
from ultralytics import YOLO
from function.sort_charater import sort_by_rows
from function.process_frame import process_frame
import os

parser = argparse.ArgumentParser(description="License Plate Recognition with YOLOv8")
parser.add_argument("--image", type=str, help="Path to input image")
parser.add_argument("--video", type=str, help="Path to input video")
parser.add_argument("--webcam", action="store_true", help="Use webcam for input")
parser.add_argument("--output", type=str, default="detected_result", help="Path to save output file (without extension)")
args = parser.parse_args()

if not args.image and not args.video and not args.webcam:
    print("❌ Vui lòng cung cấp đường dẫn ảnh, video hoặc sử dụng webcam!")
    parser.print_help()
    exit()

plate_detector = YOLO("model/LP_detector.pt") 
char_detector = YOLO("model/LP_ocr.pt")  

if args.image:
    frame = cv2.imread(args.image)
    if frame is None:
        raise ValueError("Không thể đọc ảnh. Kiểm tra đường dẫn!")
    processed_frame = process_frame(frame, plate_detector, char_detector)
    
    # Lưu ảnh với phần mở rộng .png
    output_path = f"{args.output}.jpg"
    cv2.imwrite(output_path, processed_frame)
    print(f"✅ Ảnh kết quả đã được lưu tại {output_path}")
    cv2.imshow('License Plate Recognition', processed_frame)
    cv2.waitKey()
    cv2.destroyAllWindows()
    
elif args.video:
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise ValueError("Không thể đọc video. Kiểm tra đường dẫn!")
    
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Lưu video với phần mở rộng .mp4
    output_path = f"{args.output}.mp4"
    
    # Tạo video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        processed_frame = process_frame(frame, plate_detector, char_detector)
        out.write(processed_frame)
        
        # Hiển thị frame
        cv2.imshow('License Plate Recognition', processed_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"✅ Video kết quả đã được lưu tại {output_path}")

elif args.webcam:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise ValueError("❌ Không thể mở camera!")
    print("✅ Đã kết nối camera thành công!")
    print("Nhấn 'q' để thoát")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Không thể lấy khung hình từ camera!")
            break

        processed_frame = process_frame(frame, plate_detector, char_detector)
        
        # Hiển thị frame
        cv2.imshow('License Plate Recognition', processed_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    cap.release()
    cv2.destroyAllWindows()
  
    