import os
import uuid
import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from detector_improved import AgeGenderDetectorImproved

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

detector = AgeGenderDetectorImproved()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def process_image(image_path):
    frame = cv2.imread(image_path)
    if frame is None:
        return None, []
    result_img, results = detector.process_image(frame)
    output_filename = f"result_{uuid.uuid4().hex[:8]}.jpg"
    output_path = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)
    cv2.imwrite(output_path, result_img)
    return output_filename, results

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/detect', methods=['POST'])
def detect():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    if not allowed_file(file.filename):
        return jsonify({'error': 'File type not allowed'}), 400
    filename = secure_filename(file.filename)
    ext = filename.rsplit('.', 1)[1].lower()
    temp_filename = f"input_{uuid.uuid4().hex[:8]}.{ext}"
    temp_path = os.path.join(app.config['UPLOAD_FOLDER'], temp_filename)
    file.save(temp_path)
    output_filename, results = process_image(temp_path)
    os.remove(temp_path)
    if output_filename is None:
        return jsonify({'error': 'Failed to process image'}), 500
    result_url = f'/uploads/{output_filename}'
    return jsonify({
        'success': True,
        'faces': len(results),
        'results': results,
        'result_image': result_url
    })

@app.route('/api/detect_base64', methods=['POST'])
def detect_base64():
    data = request.get_json()
    if not data or 'image' not in data:
        return jsonify({'error': 'No image data'}), 400
    import base64
    img_data = base64.b64decode(data['image'].split(',')[1] if ',' in data['image'] else data['image'])
    nparr = np.frombuffer(img_data, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        return jsonify({'error': 'Invalid image data'}), 400
    temp_path = os.path.join(app.config['UPLOAD_FOLDER'], f"webcam_{uuid.uuid4().hex[:8]}.jpg")
    cv2.imwrite(temp_path, frame)
    output_filename, results = process_image(temp_path)
    os.remove(temp_path)
    if output_filename is None:
        return jsonify({'error': 'Failed to process image'}), 500
    result_url = f'/uploads/{output_filename}'
    return jsonify({
        'success': True,
        'faces': len(results),
        'results': results,
        'result_image': result_url
    })

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    print("=" * 60)
    print("  Advanced Age & Gender Detection - Web App")
    print("  Open browser at: http://127.0.0.1:5000")
    print("=" * 60)
    app.run(debug=False, host='127.0.0.1', port=5000)
