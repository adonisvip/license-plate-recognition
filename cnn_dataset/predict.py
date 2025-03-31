import cv2
import numpy as np
import tensorflow as tf
from config import CHARS, IMG_SIZE, MODEL_PATH

# Load model
model = tf.keras.models.load_model(MODEL_PATH)

def predict_character(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, IMG_SIZE) / 255.0  # Chuẩn hóa
    img = img.reshape(1, IMG_SIZE[0], IMG_SIZE[1], 1)
    
    prediction = model.predict(img)
    predicted_label = np.argmax(prediction)
    return CHARS[predicted_label]

# Test với ảnh mới
img_path = "test_image/1_LP_sample.png"  # Cập nhật đường dẫn ảnh
print("🔍 Dự đoán ký tự:", predict_character(img_path))
