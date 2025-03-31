import cv2
import torch
import pytesseract
from ultralytics import YOLO

model = YOLO("model/last.pt")  
cap = cv2.VideoCapture(0) 

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    results = model(frame)
    
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0].item()
            
            if conf < 0.5:
                continue
            
            plate_img = frame[y1:y2, x1:x2]
            
            gray_plate = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
            text = pytesseract.image_to_string(gray_plate, config='--psm 7')
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, text.strip(), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    cv2.imshow("License Plate Detection", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
