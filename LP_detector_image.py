import argparse
import cv2
from ultralytics import YOLO
from function.sort_charater import sort_by_rows

parser = argparse.ArgumentParser(description="License Plate Recognition with YOLOv8")
parser.add_argument("--image", type=str, help="Path to input image")
parser.add_argument("--video", type=str, help="Path to input video")
parser.add_argument("--webcam", action="store_true", help="Use webcam for input")
parser.add_argument("--output", type=str, default="detected_result.png", help="Path to save output image/video")
args = parser.parse_args()

if not args.image and not args.video and not args.webcam:
    raise ValueError("Vui lòng cung cấp đường dẫn ảnh, video hoặc sử dụng webcam!")

plate_detector = YOLO("model/LP_detector.pt") 
char_detector = YOLO("model/LP_ocr.pt")  

def process_frame(frame):
    # ======== BƯỚC 1: Phát hiện biển số ========
    plate_results = plate_detector(frame)
    for plate in plate_results[0].boxes.xyxy:
        x1, y1, x2, y2 = map(int, plate.tolist())
        
        # Cắt ảnh biển số
        plate_img = frame[y1:y2, x1:x2].copy()
        cv2.imwrite("crop.jpg", plate_img)
        if plate_img is None or plate_img.size == 0:
            print("⚠ Không thể cắt biển số, bỏ qua.")
            continue

        # ======== BƯỚC 2: Phát hiện ký tự trên biển số ========
        char_results = char_detector(plate_img)
        detected_chars = []

        for box, conf, cls in zip(char_results[0].boxes.xyxy, char_results[0].boxes.conf, char_results[0].boxes.cls):
            x_min, y_min, x_max, y_max = map(int, box.tolist())
            label = char_results[0].names[int(cls)]

            if conf > 0.5:  # Lọc ký tự có độ tin cậy cao
                detected_chars.append(((x_min, y_min, y_max), label))  # Lưu (x_min, y_min, y_max)

        # ======== SẮP XẾP KÝ TỰ ========
        # Nhận diện biển số đầy đủ
        plate_text = sort_by_rows(detected_chars)
        print("Plate text:", plate_text)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(frame, plate_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
    
    return frame

if args.image:
    # Xử lý ảnh
    image = cv2.imread(args.image)
    if image is None:
        raise ValueError("Không thể đọc ảnh. Kiểm tra đường dẫn!")
    
    processed_image = process_frame(image)
    cv2.imwrite(args.output, processed_image)
    print(f"✅ Ảnh kết quả đã được lưu tại {args.output}")
    cv2.imshow('frame', processed_image)
    cv2.waitKey()
    cv2.destroyAllWindows()

elif args.video:
    # Xử lý video từ file
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise ValueError("Không thể đọc video. Kiểm tra đường dẫn!")

    # Lấy thông tin video
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Tạo video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output, fourcc, fps, (frame_width, frame_height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        processed_frame = process_frame(frame)
        out.write(processed_frame)
        
        # Hiển thị frame
        cv2.imshow('frame', processed_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Giải phóng tài nguyên
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"✅ Video kết quả đã được lưu tại {args.output}")

elif args.webcam:
    # Xử lý video từ webcam
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

        processed_frame = process_frame(frame)
        
        # Hiển thị frame
        cv2.imshow('License Plate Recognition', processed_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Giải phóng tài nguyên
    cap.release()
    cv2.destroyAllWindows()