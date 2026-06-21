import sys, os, io, json
sys.path.insert(0, '.')
from app import app

with app.test_client() as c:
    img_path = os.path.join(os.getcwd(), 'man1.jpg')
    with open(img_path, 'rb') as f:
        data = {'image': (io.BytesIO(f.read()), 'man1.jpg')}
        r = c.post('/api/detect', data=data, content_type='multipart/form-data')
        result = json.loads(r.data)
        print(f"POST /api/detect: {r.status_code}")
        print(f"Faces: {result['faces']}")
        for face in result['results']:
            print(f"  ID-{face['id']}: {face['gender']} ({face['gender_conf']}), Age={face['age']} ({face['age_conf']})")
print("API: WORKING!")
