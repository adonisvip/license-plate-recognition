from ultralytics import YOLO
model = YOLO("yolov8s.pt")

#yolo train model=yolov8n.pt data=dataset.yaml epochs=100 imgsz=1280  batch=16