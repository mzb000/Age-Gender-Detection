<div align="center">

# 🧠 Age & Gender Detection using Deep Learning

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python)](https://python.org)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.13.0-green?style=for-the-badge&logo=opencv)](https://opencv.org)
[![Flask](https://img.shields.io/badge/Flask-3.1.3-black?style=for-the-badge&logo=flask)](https://flask.palletsprojects.com)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)
[![GitHub repo size](https://img.shields.io/github/repo-size/mzb000/Age-Gender-Detection?style=for-the-badge&logo=github)](https://github.com/mzb000/Age-Gender-Detection)
[![GitHub last commit](https://img.shields.io/github/last-commit/mzb000/Age-Gender-Detection?style=for-the-badge&logo=git)](https://github.com/mzb000/Age-Gender-Detection)
[![GitHub stars](https://img.shields.io/github/stars/mzb000/Age-Gender-Detection?style=for-the-badge&logo=github)](https://github.com/mzb000/Age-Gender-Detection)
[![GitHub forks](https://img.shields.io/github/forks/mzb000/Age-Gender-Detection?style=for-the-badge&logo=github)](https://github.com/mzb000/Age-Gender-Detection)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg?style=for-the-badge)](https://github.com/mzb000/Age-Gender-Detection)

**Real-time Age & Gender Detection** — Multi-face tracking with YuNet, OpenCV DNN, and Flask web UI.

</div>

---

## ✨ Features

- ✅ **Multi-Face Detection** — Detects multiple faces simultaneously using YuNet (OpenCV Zoo)
- ✅ **Face Alignment** — Eye-based alignment for better accuracy
- ✅ **Real-time Tracking** — CentroidTracker assigns unique IDs to each face across frames
- ✅ **Age Prediction** — 8 age ranges: (0-2), (4-6), (8-12), (15-20), (25-32), (38-43), (48-53), (60-100)
- ✅ **Gender Prediction** — Male / Female classification
- ✅ **Web UI** — Flask-based dark theme interface with drag & drop and webcam support
- ✅ **CLI Mode** — Command-line interface for images and webcam
- ✅ **High Confidence** — CLAHE preprocessing + brightness normalization

## 🖼️ Screenshots

<!-- Add screenshots here if available -->

## 📦 Installation

```bash
# Clone the repo
git clone https://github.com/mzb000/Age-Gender-Detection.git
cd Age-Gender-Detection

# Install dependencies
pip install -r requirements.txt
```

## 🚀 Usage

### Web App (Browser UI)

```bash
python app.py
```
Open **http://127.0.0.1:5000** in your browser.

### CLI — Image Mode

```bash
python detect_advanced.py --image man1.jpg
```

### CLI — Webcam Mode

```bash
python detect_advanced.py
```

**Controls:** `q` = quit | `s` = save snapshot

## 🗂️ Project Structure

```
├── app.py                      # Flask web application
├── detect_advanced.py          # CLI version
├── detector_improved.py        # Improved detection module (YuNet + Alignment)
├── centroid_tracker.py         # Face tracking engine
├── templates/index.html        # Web UI (Bootstrap 5 Dark Theme)
├── requirements.txt            # Python dependencies
├── run.bat                     # One-click launcher
├── face_detection_yunet_2023mar.onnx  # YuNet face detection model
├── age_deploy.prototxt         # Age model architecture
├── age_net.caffemodel          # Age model weights
├── gender_deploy.prototxt      # Gender model architecture
└── gender_net.caffemodel       # Gender model weights
```

## 📊 Accuracy Improvements

| Technique | Improvement |
|-----------|-------------|
| YuNet Face Detection | More accurate face bounding boxes |
| Face Alignment | Corrects head pose before prediction |
| CLAHE Preprocessing | Reduces lighting variation effects |
| Smart Padding | Optimal face context extraction |
| Temporal Smoothing | Stable predictions across video frames |

## ⚙️ How It Works

1. **Face Detection** — YuNet (OpenCV Zoo) detects faces in the frame
2. **Face Alignment** — Eye landmarks are used to align the face
3. **Preprocessing** — CLAHE + brightness normalization improves input quality
4. **Prediction** — Caffe models classify gender and age
5. **Tracking** — CentroidTracker assigns persistent IDs to faces
6. **Display** — Results shown with colored bounding boxes and labels

## 📝 License

This project is licensed under the MIT License.

## 🙏 Credits

- Pre-trained models by [Tal Hassner and Gil Levi](https://talhassner.github.io/home/projects/Adience/Adience-data.html)
- YuNet model from [OpenCV Zoo](https://github.com/opencv/opencv_zoo)
- Original project inspiration from [ProjectGurukul](https://projectgurukul.org)

---
<div align="center">
Made with ❤️ by <a href="https://github.com/mzb000">mzb000</a>
</div>
