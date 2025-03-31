import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# Định nghĩa các ký tự có thể nhận diện (số + chữ cái)
CHARS = "0123456789ABCDEFGHIJKLMNPQRSTUVWXYZ"
CHAR_DICT = {char: i for i, char in enumerate(CHARS)}

# Đọc ảnh từ dataset
def load_data(data_dir, img_size=(28,28)):
    images, labels = [], []
    for label in os.listdir(data_dir):
        label_path = os.path.join(data_dir, label)
        if os.path.isdir(label_path):
            for img_file in os.listdir(label_path):
                img_path = os.path.join(label_path, img_file)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                img = cv2.resize(img, img_size)
                img = img / 255.0  # Chuẩn hóa pixel về [0,1]
                images.append(img)
                labels.append(CHAR_DICT[label])
    return np.array(images), np.array(labels)

# Load dataset
data_dir = "train"  # Cập nhật đường dẫn đúng
X, y = load_data(data_dir)

# Chia train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Reshape ảnh để phù hợp với mô hình CNN
X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)

# One-hot encode labels
y_train = to_categorical(y_train, num_classes=len(CHARS))
y_test = to_categorical(y_test, num_classes=len(CHARS))
