import cv2
import time
import argparse
import os
from centroid_tracker import CentroidTracker
from detector_improved import AgeGenderDetectorImproved

tracker = CentroidTracker(max_disappeared=30)
detector = AgeGenderDetectorImproved()
frame_count = 0

def process_frame(frame):
    global frame_count
    frame_count += 1
    use_temporal = frame_count > 5

    result_img, raw_results = detector.process_image(frame, use_temporal=use_temporal)

    bboxes = [(r['box'][0], r['box'][1], r['box'][2], r['box'][3]) for r in raw_results]
    objects, colors = tracker.update(bboxes)

    for obj_id, (centroid, (x1, y1, x2, y2)) in objects.items():
        matching = [r for r in raw_results if abs(r['box'][0] - x1) < 20 and abs(r['box'][1] - y1) < 20]
        if matching:
            r = matching[0]
            color = colors[obj_id]
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label_id = f"ID-{obj_id}"
            label_info = f"{r['gender']}, {r['age']}"
            cv2.rectangle(frame, (x1, y1 - 50), (x2, y1), color, -1)
            cv2.putText(frame, label_id, (x1 + 5, y1 - 32), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
            cv2.putText(frame, label_info, (x1 + 5, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return frame, len(objects)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", help="path to input image (leave empty for webcam)")
    args = ap.parse_args()

    if args.image:
        if not os.path.exists(args.image):
            print(f"[ERROR] Image not found: {args.image}")
            return
        frame = cv2.imread(args.image)
        if frame is None:
            print("[ERROR] Could not read image")
            return
        frame, count = process_frame(frame)
        print(f"[INFO] Detected {count} face(s)")
        cv2.imshow("Improved Age & Gender Detection (YuNet + Alignment)", frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("[ERROR] Could not open webcam")
            return
        print("[INFO] Webcam started. Press 'q' to quit, 's' to save snapshot")

        fps_start = time.time()
        fps_frames = 0
        fps = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame, count = process_frame(frame)

            fps_frames += 1
            if time.time() - fps_start >= 1.0:
                fps = fps_frames
                fps_frames = 0
                fps_start = time.time()

            cv2.putText(frame, f"FPS: {fps} | Faces: {count}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.imshow("Improved Age & Gender Detection (YuNet + Alignment)", frame)

            k = cv2.waitKey(1) & 0xFF
            if k == ord('q'):
                break
            elif k == ord('s'):
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = f"snapshot_{timestamp}.jpg"
                cv2.imwrite(filename, frame)
                print(f"[INFO] Snapshot saved: {filename}")

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
