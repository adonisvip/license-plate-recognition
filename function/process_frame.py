from function.sort_charater import sort_by_rows
import cv2

def process_frame(frame, plate_detector, char_detector):
    # ======== BƯỚC 1: Phát hiện biển số ========
    plate_results = plate_detector(frame)
    for plate in plate_results[0].boxes.xyxy:
        x1, y1, x2, y2 = map(int, plate.tolist())
        
        # Cắt ảnh biển số
        plate_img = frame[y1:y2, x1:x2].copy()
        cv2.imwrite("result/crop.jpg", plate_img)
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
