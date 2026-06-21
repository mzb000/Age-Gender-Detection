import cv2
import numpy as np
import os
from collections import deque

BASE = "D:/ML Learning/machine-learning-projects/gender-age-detection-advanced"

class AgeGenderDetectorImproved:
    def __init__(self):
        self.ageProto = os.path.join(BASE, "age_deploy.prototxt")
        self.ageModel = os.path.join(BASE, "age_net.caffemodel")
        self.genderProto = os.path.join(BASE, "gender_deploy.prototxt")
        self.genderModel = os.path.join(BASE, "gender_net.caffemodel")
        self.yunet_model = os.path.join(BASE, "face_detection_yunet_2023mar.onnx")

        self.MODEL_MEAN = (78.4263377603, 87.7689143744, 114.895847746)
        self.AGE_LIST = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
        self.GENDER_LIST = ['Male', 'Female']
        self.COLOR_POOL = [
            (0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0),
            (255, 0, 255), (0, 255, 255), (128, 255, 0), (255, 128, 0),
        ]

        self.ageNet = cv2.dnn.readNet(self.ageModel, self.ageProto)
        self.genderNet = cv2.dnn.readNet(self.genderModel, self.genderProto)

        self.face_detector = cv2.FaceDetectorYN.create(
            self.yunet_model, "", (320, 320), 0.8, 0.4, 5000
        )
        self.fallback_faceNet = cv2.dnn.readNet(
            os.path.join(BASE, "opencv_face_detector_uint8.pb"),
            os.path.join(BASE, "opencv_face_detector.pbtxt")
        )

        self.temporal_buffer = {}
        self.gender_ema = {}
        self.age_ema = {}

    def apply_clahe(self, face_bgr):
        lab = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
        l_eq = clahe.apply(l)
        lab_eq = cv2.merge([l_eq, a, b])
        face_eq = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)
        blended = cv2.addWeighted(face_bgr, 0.6, face_eq, 0.4, 0)
        return blended

    def normalize_brightness(self, face_bgr):
        mean_brightness = np.mean(face_bgr)
        target = 128
        if mean_brightness < 80 or mean_brightness > 180:
            alpha = target / max(mean_brightness, 1)
            face_norm = cv2.convertScaleAbs(face_bgr, alpha=alpha, beta=0)
            return cv2.addWeighted(face_bgr, 0.5, face_norm, 0.5, 0)
        return face_bgr.copy()

    def smart_padding(self, x1, y1, x2, y2, img_w, img_h, pad=0.2):
        bw, bh = x2 - x1, y2 - y1
        px, py = int(bw * pad), int(bh * pad)
        x1_e = max(0, x1 - px)
        y1_e = max(0, y1 - py)
        x2_e = min(img_w, x2 + px)
        y2_e = min(img_h, y2 + py)
        return x1_e, y1_e, x2_e, y2_e

    def detect_faces_yunet(self, image_bgr):
        h, w = image_bgr.shape[:2]
        self.face_detector.setInputSize((w, h))
        _, faces = self.face_detector.detect(image_bgr)
        results = []
        if faces is not None:
            for face in faces:
                x, y, wf, hf = int(face[0]), int(face[1]), int(face[2]), int(face[3])
                conf = float(face[14])
                if conf > 0.7:
                    landmarks = None
                    if face.shape[0] >= 15:
                        landmarks = {
                            'left_eye': (int(face[4]), int(face[5])),
                            'right_eye': (int(face[6]), int(face[7])),
                            'nose': (int(face[8]), int(face[9])),
                            'mouth_left': (int(face[10]), int(face[11])),
                            'mouth_right': (int(face[12]), int(face[13])),
                        }
                    results.append({
                        'box': (x, y, x + wf, y + hf),
                        'confidence': conf,
                        'keypoints': landmarks
                    })
        return results

    def detect_faces_fallback(self, image_bgr):
        h, w = image_bgr.shape[:2]
        blob = cv2.dnn.blobFromImage(image_bgr, 1.0, (300, 300), [104, 117, 123], swapRGB=False)
        self.fallback_faceNet.setInput(blob)
        detections = self.fallback_faceNet.forward()
        results = []
        for i in range(detections.shape[2]):
            conf = float(detections[0, 0, i, 2])
            if conf > 0.7:
                x1 = int(detections[0, 0, i, 3] * w)
                y1 = int(detections[0, 0, i, 4] * h)
                x2 = int(detections[0, 0, i, 5] * w)
                y2 = int(detections[0, 0, i, 6] * h)
                results.append({
                    'box': (x1, y1, x2, y2),
                    'confidence': conf,
                    'keypoints': None
                })
        return results

    def align_face(self, image_bgr, keypoints):
        left_eye = np.array(keypoints['left_eye'], dtype=np.float32)
        right_eye = np.array(keypoints['right_eye'], dtype=np.float32)
        eye_center = (left_eye + right_eye) / 2.0
        dx = right_eye[0] - left_eye[0]
        dy = right_eye[1] - left_eye[1]
        angle = np.degrees(np.arctan2(dy, dx))
        M = cv2.getRotationMatrix2D(tuple(eye_center.astype(float)), angle, 1.0)
        h, w = image_bgr.shape[:2]
        aligned = cv2.warpAffine(image_bgr, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        return aligned

    def predict_single(self, face_bgr):
        face_pp = self.normalize_brightness(face_bgr)
        face_pp = self.apply_clahe(face_pp)

        blob = cv2.dnn.blobFromImage(face_pp, 1.0, (227, 227), self.MODEL_MEAN, swapRB=False)

        self.genderNet.setInput(blob)
        gender_pred = self.genderNet.forward()
        gender = self.GENDER_LIST[int(gender_pred[0].argmax())]
        gender_conf = float(gender_pred[0].max())

        self.ageNet.setInput(blob)
        age_pred = self.ageNet.forward()
        age_scores = age_pred[0]
        age_idx = int(np.argmax(age_scores))
        age = self.AGE_LIST[age_idx]
        age_conf = float(age_scores[age_idx])

        return gender, gender_conf, age, age_conf

    def ema_smooth(self, face_id, gender, gender_conf, age, age_conf, alpha=0.6):
        if face_id not in self.gender_ema:
            self.gender_ema[face_id] = {'Male': 0.0, 'Female': 0.0}
            self.age_ema[face_id] = {}
            for a in self.AGE_LIST:
                self.age_ema[face_id][a] = 0.0

        for g in self.gender_ema[face_id]:
            self.gender_ema[face_id][g] *= (1 - alpha)
        self.gender_ema[face_id][gender] += alpha * gender_conf

        for a in self.age_ema[face_id]:
            self.age_ema[face_id][a] *= (1 - alpha)
        self.age_ema[face_id][age] += alpha * age_conf

        smoothed_gender = max(self.gender_ema[face_id], key=self.gender_ema[face_id].get)
        smoothed_age = max(self.age_ema[face_id], key=self.age_ema[face_id].get)
        smoothed_gender_conf = self.gender_ema[face_id][smoothed_gender]
        smoothed_age_conf = self.age_ema[face_id][smoothed_age]

        return smoothed_gender, smoothed_gender_conf, smoothed_age, smoothed_age_conf

    def process_image(self, image_bgr, use_temporal=False, face_ids=None):
        h, w = image_bgr.shape[:2]

        faces = self.detect_faces_yunet(image_bgr)
        if not faces:
            faces = self.detect_faces_fallback(image_bgr)

        results = []
        for idx, face_data in enumerate(faces):
            x1, y1, x2, y2 = face_data['box']
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 - x1 < 10 or y2 - y1 < 10:
                continue

            color = self.COLOR_POOL[idx % len(self.COLOR_POOL)]

            if face_data['keypoints']:
                aligned = self.align_face(image_bgr, face_data['keypoints'])
                sx1, sy1, sx2, sy2 = self.smart_padding(x1, y1, x2, y2, w, h, pad=0.25)
                face_crop = aligned[sy1:sy2, sx1:sx2]
            else:
                sx1, sy1, sx2, sy2 = self.smart_padding(x1, y1, x2, y2, w, h, pad=0.2)
                face_crop = image_bgr[sy1:sy2, sx1:sx2]

            if face_crop.size == 0:
                continue

            gender, gender_conf, age, age_conf = self.predict_single(face_crop)

            if use_temporal and face_ids and idx < len(face_ids):
                fid = face_ids[idx]
                gender, gender_conf, age, age_conf = self.ema_smooth(fid, gender, gender_conf, age, age_conf)

            cv2.rectangle(image_bgr, (x1, y1), (x2, y2), color, 2)
            label_id = f"ID-{face_ids[idx] if face_ids else idx}"
            label_info = f"{gender}, {age}"

            cv2.rectangle(image_bgr, (x1, y1 - 50), (x2, y1), color, -1)
            cv2.putText(image_bgr, label_id, (x1 + 5, y1 - 32),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
            cv2.putText(image_bgr, label_info, (x1 + 5, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            results.append({
                'id': face_ids[idx] if face_ids else idx,
                'gender': gender,
                'gender_conf': round(gender_conf, 3),
                'age': age,
                'age_conf': round(age_conf, 3),
                'box': [int(x1), int(y1), int(x2), int(y2)],
                'color': (int(color[0]), int(color[1]), int(color[2]))
            })

        return image_bgr, results
