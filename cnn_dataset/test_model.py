import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

ocr_model = load_model("weight.h5")

image_path = "../test_image/1_LP_sample.png"  # Thay bằng ảnh biển số cần nhận diện
image = cv2.imread(image_path)

if image is None:
    print("Lỗi: Không thể đọc ảnh!")
    exit()
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cropped_plate_resized = cv2.resize(gray_image, (128, 64))  # Resize về đúng kích thước
normalized_plate = cropped_plate_resized.astype("float32") / 255.0  # Chuẩn hóa ảnh
img_array = img_to_array(normalized_plate)
img_array = np.expand_dims(img_array, axis=0)  # Thêm batch dimension

prediction = ocr_model.predict(img_array)
plate_number = "".join([chr(np.argmax(p) + 48) for p in prediction])  # Chuyển thành ký tự

print(f"📸 Biển số nhận diện: {plate_number}")

cv2.putText(image, plate_number, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
cv2.imshow("License Plate OCR", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
