import sys, os, cv2, time
sys.path.insert(0, '.')
from detector_improved import AgeGenderDetectorImproved

d = AgeGenderDetectorImproved()
for fname in ['man1.jpg', 'girl1.jpg', 'kid1.jpg']:
    img = cv2.imread(fname)
    if img is None:
        print(f"{fname}: FAIL read")
        continue
    start = time.time()
    res_img, results = d.process_image(img)
    elapsed = time.time() - start
    for r in results:
        print(f"{fname}: ID-{r['id']}: {r['gender']} ({r['gender_conf']:.3f}), Age={r['age']} ({r['age_conf']:.3f}) [{elapsed:.2f}s]")
print("DONE")
