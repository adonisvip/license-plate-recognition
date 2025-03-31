import os

# Danh sách ký tự có thể nhận diện (0-9, A-Z)
CHARS = "0123456789ABCDEFGHIJKLMNPQRSTUVWXYZ"
NUM_CLASSES = len(CHARS)
CHAR_DICT = {char: i for i, char in enumerate(CHARS)}

# Kích thước ảnh đầu vào
IMG_SIZE = (28, 28)

# Đường dẫn dataset
DATASET_PATH = "train"  # Thay đổi nếu cần
MODEL_PATH = "character_recognition_model.h5"