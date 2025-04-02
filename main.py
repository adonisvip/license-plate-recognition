import argparse
from ultralytics import YOLO
from function.process import process_image, process_video
import sys

def main():
    # Thiết lập tham số dòng lệnh
    parser = argparse.ArgumentParser(description="License Plate Recognition with YOLOv8")
    parser.add_argument("--image", type=str, help="Path to input image")
    parser.add_argument("--video", type=str, help="Path to input video")
    parser.add_argument("--webcam", action="store_true", help="Use webcam for real-time detection")
    parser.add_argument("--output", type=str, default="detected_result", help="Path to save output file (without extension)")
    args = parser.parse_args()

    # Kiểm tra tham số đầu vào
    if not args.image and not args.video and not args.webcam:
        print("❌ Vui lòng cung cấp đường dẫn ảnh, video hoặc sử dụng webcam!")
        parser.print_help()
        sys.exit(1)

    # Kiểm tra xung đột tham số
    if args.webcam and (args.image or args.video):
        print("❌ Không thể sử dụng webcam cùng lúc với ảnh hoặc video!")
        sys.exit(1)

    # Khởi tạo các model detector
    try:
        plate_detector = YOLO("model/LP_detector.pt")
        char_detector = YOLO("model/LP_ocr.pt")
    except Exception as e:
        print(f"❌ Lỗi khi khởi tạo model: {str(e)}")
        sys.exit(1)

    try:
        if args.image:
            # Xử lý ảnh
            output_path = f"result/{args.output}.jpg"
            process_image(args.image, output_path, plate_detector, char_detector)
            
        elif args.video:
            # Xử lý video từ file
            output_path = f"result/{args.output}.mp4"
            process_video(args.video, output_path, plate_detector, char_detector)
            
        elif args.webcam:
            # Xử lý video từ webcam (không lưu output)
            process_video(None, None, plate_detector, char_detector, save_output=False)
            
    except Exception as e:
        print(f"❌ Lỗi: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
  
    
