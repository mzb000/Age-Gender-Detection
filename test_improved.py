import sys, os, cv2, time
sys.path.insert(0, '.')
from detector_improved import AgeGenderDetectorImproved

detector = AgeGenderDetectorImproved()

for img_name in ['man1.jpg', 'girl1.jpg', 'kid1.jpg']:
    img = cv2.imread(img_name)
    if img is None:
        print(f'{img_name}: FAIL - could not read')
        continue
    start = time.time()
    result_img, results = detector.process_image(img)
    elapsed = time.time() - start
    print(f'\n=== {img_name} ({elapsed:.2f}s) ===')
    if not results:
        print('  No faces detected')
    else:
        for r in results:
            print(f'  ID-{r["id"]}: {r["gender"]} ({r["gender_conf"]:.3f}), Age={r["age"]} ({r["age_conf"]:.3f}), Box={r["box"]}')
