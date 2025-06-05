import os
from flask import Flask, request, render_template, jsonify, send_from_directory
import cv2
import numpy as np
from tensorflow import keras
from preprocessing1 import grayScaleConversion, noiseRemoval, imageEnhancement, segmentation, segment
from hairRemoval import dullRazor_single
import uuid

app = Flask(__name__)

UPLOAD_FOLDER = 'inputs'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

try:
    cnn = keras.models.load_model('cnn.h5')
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")

def preprocess_image(image_path, unique_id):
    # Read the image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Error: Unable to read image {image_path}")

    # Grayscale conversion
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_image_path = os.path.join(UPLOAD_FOLDER, f'gray_image_{unique_id}.png')
    cv2.imwrite(gray_image_path, img_gray)

    # DullRazor
    img_dullrazor = dullRazor_single(img_gray)
    dullrazor_image_path = os.path.join(UPLOAD_FOLDER, f'dullrazor_image_{unique_id}.png')
    cv2.imwrite(dullrazor_image_path, img_dullrazor)

    # Noise removal
    img_noise_removed = cv2.medianBlur(img_dullrazor, 3)
    noise_removed_image_path = os.path.join(UPLOAD_FOLDER, f'noise_removed_image_{unique_id}.png')
    cv2.imwrite(noise_removed_image_path, img_noise_removed)

    # Image enhancement
    img_enhanced = np.uint8(cv2.normalize(img_noise_removed, None, 0, 255, cv2.NORM_MINMAX))
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_enhanced = clahe.apply(img_enhanced)
    enhanced_image_path = os.path.join(UPLOAD_FOLDER, f'enhanced_image_{unique_id}.png')
    cv2.imwrite(enhanced_image_path, img_enhanced)

    # Segmentation
    img_segmented = segment(img_enhanced)
    segmented_image_path = os.path.join(UPLOAD_FOLDER, f'segmented_image_{unique_id}.png')
    cv2.imwrite(segmented_image_path, img_segmented)

    # Resize and normalize the image
    resized_img = cv2.resize(img_segmented, (224, 224))
    normalized_img = resized_img / 255.0
    input_img = np.expand_dims(normalized_img, axis=0)
    input_img = np.expand_dims(input_img, axis=-1)  # Add channel dimension

    return input_img, {
        'gray': gray_image_path,
        'dullrazor': dullrazor_image_path,
        'noise_removed': noise_removed_image_path,
        'enhanced': enhanced_image_path,
        'segmented': segmented_image_path
    }

@app.route('/')
def upload_form():
    return render_template('main.html', message='')

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return "No image file provided", 400

    image = request.files['image']
    allowed_extensions = {'png', 'jpg', 'jpeg', 'gif'}
    if image.filename == '':
        return render_template('main.html', message="Empty file")

    if not '.' in image.filename or image.filename.split('.')[-1].lower() not in allowed_extensions:
        return render_template('main.html', message="Invalid file type")

    unique_id = str(uuid.uuid4())
    filename = f"{unique_id}_{image.filename}"
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    image.save(filepath)
    
    return jsonify({'filename': filename})

@app.route('/preprocess', methods=['POST'])
def preprocess():
    data = request.json
    filename = data['filename']
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    unique_id = filename.split('_')[0]

    try:
        input_img, images = preprocess_image(filepath, unique_id)
        return jsonify(images)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/classify', methods=['POST'])
def classify():
    data = request.json
    filename = data['filename']
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    unique_id = filename.split('_')[0]

    try:
        input_img, _ = preprocess_image(filepath, unique_id)
        res = cnn.predict(input_img)
        res_class = np.argmax(res)
        if res_class == 0:
            message = 'The image is non-cancerous'
        else:
            message = 'The image is cancerous'

        return jsonify({'message': message})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/inputs/<filename>')
def send_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

if __name__ == '__main__':
    app.run(debug=True)
