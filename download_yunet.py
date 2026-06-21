import cv2
import os

base = "D:/ML Learning/machine-learning-projects/gender-age-detection-advanced"
model_path = os.path.join(base, "face_detection_yunet_2023mar.onnx")

if not os.path.exists(model_path):
    print("Downloading YuNet face detection model...")
    url = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
    import urllib.request
    urllib.request.urlretrieve(url, model_path)
    print(f"Downloaded to: {model_path}")
else:
    print(f"Model already exists: {model_path}")
