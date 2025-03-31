# from ultralytics import YOLO
# import cv2

# model = YOLO("last.pt")
# image = cv2.imread("4.jpg")
# results = model(image, show=True)
# print(results[0].boxes)
# cv2.waitKey(0)
# cropped_plate_gray = cv2.cvtColor(cropped_plate, cv2.COLOR_BGR2GRAY)
# _, cropped_plate_thresh = cv2.threshold(cropped_plate_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)


from ultralytics import YOLO
import cv2

model = YOLO("Char_best.pt")
image = cv2.imread("1_LP_sample.png")
results = model(image, show=True)
print(results[0].boxes)
cv2.waitKey(0)

