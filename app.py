import os
import tensorflow as tf
import numpy as np
import uuid
from flask import Flask, render_template, request, jsonify, url_for
from werkzeug.utils import secure_filename
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import Input
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
MODEL_PATH = 'classifymodel.h5'
IMAGE_SIZE = (128, 128)
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file sizes

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_trained_model():
    if os.path.exists(MODEL_PATH):
        print("Model loaded successfully!")
        return load_model(MODEL_PATH)
    print("No trained model found!")
    return None

model = load_trained_model()
class_mapping = {}  # where to store class name to index mapping

def load_data():
    global class_mapping
    image_data, labels = [], []
    class_mapping = {}
    
    for file in os.listdir(UPLOAD_FOLDER):
        if file.endswith(tuple(ALLOWED_EXTENSIONS)):
            parts = file.split('_')
            if len(parts) < 2:
                continue
            label = parts[0]
            if label not in class_mapping:
                class_mapping[label] = len(class_mapping)
            img_path = os.path.join(UPLOAD_FOLDER, file)
            img = load_img(img_path, target_size=IMAGE_SIZE)
            img_array = img_to_array(img) / 255.0
            image_data.append(img_array)
            labels.append(class_mapping[label])
    
    return np.array(image_data), np.array(labels)

def create_model(num_classes):
    model = Sequential([
        Input(shape=(*IMAGE_SIZE, 3)),
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload_samples', methods=['POST'])
def upload_samples():
    if 'images' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    files = request.files.getlist('images')
    class_name = request.form.get('class_name')
    
    if not class_name:
        return jsonify({'error': 'No class name provided'}), 400
    
    saved_files = []
    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(f"{class_name}_{uuid.uuid4()}.{file.filename.rsplit('.', 1)[1].lower()}")
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            saved_files.append(filename)
    
    return jsonify({
        'message': f'Successfully uploaded {len(saved_files)} files',
        'files': saved_files
    })

@app.route('/train', methods=['POST'])
def train():
    global model, class_mapping
    
    image_data, labels = load_data()
    if len(class_mapping) < 2:
        return jsonify({'error': 'Need at least 2 classes for training'}), 400
    
    num_classes = len(class_mapping)
    labels_categorical = to_categorical(labels, num_classes=num_classes)
    
    X_train, X_val, y_train, y_val = train_test_split(
        image_data, labels_categorical, test_size=0.2, random_state=42
    )
    
    model = create_model(num_classes)
    early_stopping = EarlyStopping(monitor='val_loss', patience=5)
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=20,
        callbacks=[early_stopping]
    )
    
    model.save(MODEL_PATH)
    
    return jsonify({
        'message': 'Training complete',
        'class_mapping': class_mapping
    })

@app.route('/classify', methods=['POST'])
def classify():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if not file or not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file'}), 400
    
    if not model:
        return jsonify({'error': 'Model not trained yet'}), 400
    
    try:
        # save tempo file
        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], 'temp_predict.jpg')
        file.save(temp_path)
        
        # load && process the image
        img = load_img(temp_path, target_size=IMAGE_SIZE)
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0
        
        # where to make prediction
        prediction = model.predict(img_array)
        predicted_class_index = np.argmax(prediction[0])
        confidence = float(prediction[0][predicted_class_index])
        
        # get class from index
        predicted_class = [k for k, v in class_mapping.items() if v == predicted_class_index][0]
        
        # clean tempo file
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        return jsonify({
            'class': predicted_class,
            'confidence': confidence
        })
    
    except Exception as e:
        # Clean up temporary file in case of error
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return jsonify({'error': str(e)}), 500

@app.route('/check_model_status')
def check_model_status():
    return jsonify({'ready': model is not None})

if __name__ == '__main__':
    app.run(debug=True)