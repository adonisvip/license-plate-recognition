import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from config import CHARS, CHAR_DICT, IMG_SIZE, NUM_CLASSES, DATASET_PATH, MODEL_PATH

def load_data(data_dir):
    images, labels = [], []
    for label in os.listdir(data_dir):
        label_path = os.path.join(data_dir, label)
        if os.path.isdir(label_path):
            for img_file in os.listdir(label_path):
                img_path = os.path.join(label_path, img_file)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                img = cv2.resize(img, IMG_SIZE) / 255.0  # Chuẩn hóa về [0,1]
                images.append(img)
                labels.append(CHAR_DICT[label])
    return np.array(images), np.array(labels)

print("📂 Loading dataset...")
X, y = load_data(DATASET_PATH)

# Chia train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Reshape ảnh để phù hợp với CNN
X_train = X_train.reshape(-1, IMG_SIZE[0], IMG_SIZE[1], 1)
X_test = X_test.reshape(-1, IMG_SIZE[0], IMG_SIZE[1], 1)

# One-hot encode labels
y_train = to_categorical(y_train, num_classes=NUM_CLASSES)
y_test = to_categorical(y_test, num_classes=NUM_CLASSES)

# Xây dựng model CNN
print("🛠️ Building model...")
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(IMG_SIZE[0], IMG_SIZE[1], 1)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(NUM_CLASSES, activation='softmax')
])

# Compile model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train model
print("🚀 Training model...")
model.fit(X_train, y_train, epochs=15, validation_data=(X_test, y_test))

# Lưu model sau khi train xong
model.save(MODEL_PATH)
print(f"✅ Model saved at {MODEL_PATH}")
