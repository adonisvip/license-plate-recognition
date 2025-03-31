import cv2
import numpy as np
from tensorflow.keras.models import load_model
from train import *

def predict_character(image_path, model, class_indices):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (28, 28))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    predictions = model.predict(img)
    class_id = np.argmax(predictions, axis=1)[0]

    inv_class_indices = {v: k for k, v in class_indices.items()}
    character = inv_class_indices[class_id]
    
    return character

model = load_model('license_plate_character_model.h5')

# Sử dụng model
character = predict_character('test.jpg', model, train_generator.class_indices)
print("Ký tự dự đoán:", character)
