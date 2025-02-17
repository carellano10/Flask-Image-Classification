from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2
import json

app = Flask(__name__)

# Configuration constants
IMG_SIZE = (128, 128)
BATCH_SIZE = 16
UPLOAD_FOLDER = 'uploads'
DATASET_FOLDER = 'dataset'
MODEL_FOLDER = 'model'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MIN_SAMPLES_PER_CLASS = 10

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['DATASET_FOLDER'] = DATASET_FOLDER
app.config['MODEL_FOLDER'] = MODEL_FOLDER

# Create necessary directories
for folder in [UPLOAD_FOLDER, DATASET_FOLDER, MODEL_FOLDER]:
    os.makedirs(folder, exist_ok=True)

def allowed_file(filename):
    """Check if the file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def create_model(input_shape, num_classes):
    """Create an improved CNN model with better architecture and regularization"""
    model = models.Sequential([
        # First Convolutional Block
        layers.Conv2D(64, (3, 3), padding='same', activation='relu', 
                     input_shape=input_shape,
                     kernel_regularizer=regularizers.l2(0.01)),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Second Convolutional Block
        layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Third Convolutional Block
        layers.Conv2D(256, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(256, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Dense Layers
        layers.Flatten(),
        layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])

    # Compile with better learning rate and optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def check_dataset_balance():
    """Check if dataset is balanced and has minimum required samples."""
    class_counts = {}
    for class_name in os.listdir(app.config['DATASET_FOLDER']):
        class_path = os.path.join(app.config['DATASET_FOLDER'], class_name)
        if os.path.isdir(class_path):
            count = len([f for f in os.listdir(class_path) 
                        if os.path.isfile(os.path.join(class_path, f)) 
                        and allowed_file(f)])
            class_counts[class_name] = count
            
            if count < MIN_SAMPLES_PER_CLASS:
                raise ValueError(
                    f"Class '{class_name}' has only {count} samples. "
                    f"Minimum required is {MIN_SAMPLES_PER_CLASS}."
                )
    
    return class_counts

def train_model():
    """Train the model using the dataset with improved data augmentation."""
    try:
        # Check dataset balance
        class_counts = check_dataset_balance()
        
        # Enhanced data augmentation
        datagen = ImageDataGenerator(
            rescale=1./255,
            validation_split=0.2,
            rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )

        train_generator = datagen.flow_from_directory(
            app.config['DATASET_FOLDER'],
            target_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            class_mode='categorical',
            subset='training'
        )

        val_generator = datagen.flow_from_directory(
            app.config['DATASET_FOLDER'],
            target_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            class_mode='categorical',
            subset='validation'
        )

        # Save class indices
        class_indices = {v: k for k, v in train_generator.class_indices.items()}
        with open(os.path.join(app.config['MODEL_FOLDER'], 'class_indices.json'), 'w') as f:
            json.dump(class_indices, f)

        model = create_model((IMG_SIZE[0], IMG_SIZE[1], 3), len(class_indices))
        
        # Improved early stopping with more patience
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            min_delta=0.001
        )

        # Add learning rate reduction on plateau
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=3,
            min_lr=0.00001
        )

        # Train the model
        history = model.fit(
            train_generator,
            validation_data=val_generator,
            epochs=30,
            callbacks=[early_stopping, reduce_lr]
        )

        # Save training history
        with open(os.path.join(app.config['MODEL_FOLDER'], 'training_history.json'), 'w') as f:
            json.dump({
                'accuracy': [float(x) for x in history.history['accuracy']],
                'val_accuracy': [float(x) for x in history.history['val_accuracy']],
                'loss': [float(x) for x in history.history['loss']],
                'val_loss': [float(x) for x in history.history['val_loss']]
            }, f)

        # Save the model
        model.save(os.path.join(app.config['MODEL_FOLDER'], 'model.h5'))
        
        return {
            "message": "Training completed successfully",
            "class_counts": class_counts,
            "final_accuracy": float(history.history['val_accuracy'][-1]),
            "final_loss": float(history.history['val_loss'][-1])
        }
    
    except Exception as e:
        raise Exception(f"Training failed: {str(e)}")

def predict_image(image_path):
    """Predict the class of a single image with confidence thresholding."""
    try:
        model_path = os.path.join(app.config['MODEL_FOLDER'], 'model.h5')
        if not os.path.exists(model_path):
            raise FileNotFoundError("Model not found. Please train the model first.")

        model = tf.keras.models.load_model(model_path)
        
        with open(os.path.join(app.config['MODEL_FOLDER'], 'class_indices.json'), 'r') as f:
            class_indices = json.load(f)

        # Preprocess the image
        img = load_img(image_path, target_size=IMG_SIZE)
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, 0)
        img_array = img_array / 255.0

        # Make prediction
        predictions = model.predict(img_array)
        predicted_class_index = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class_index])
        
        # If confidence is too low, return "Unknown"
        if confidence < 0.3:  # Confidence threshold
            return "Unknown", confidence
            
        predicted_class_name = class_indices.get(str(predicted_class_index), "Unknown")
        
        return predicted_class_name, confidence
    
    except Exception as e:
        raise Exception(f"Prediction failed: {str(e)}")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload_samples', methods=['POST'])
def upload_samples():
    try:
        class_name = request.form.get('class_name')
        if not class_name:
            return jsonify({"error": "No class name provided"}), 400

        if 'images' not in request.files:
            return jsonify({"error": "No images uploaded"}), 400

        files = request.files.getlist('images')
        if not files:
            return jsonify({"error": "No selected files"}), 400

        # Create class directory if it doesn't exist
        class_dir = os.path.join(app.config['DATASET_FOLDER'], class_name)
        os.makedirs(class_dir, exist_ok=True)

        # Save uploaded files
        saved_files = 0
        for file in files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file.save(os.path.join(class_dir, filename))
                saved_files += 1

        return jsonify({
            "message": f"Successfully uploaded {saved_files} images for class '{class_name}'"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/train', methods=['POST'])
def train():
    try:
        result = train_model()
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/classify', methods=['POST'])
def classify():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files['file']
        if not file or not allowed_file(file.filename):
            return jsonify({"error": "Invalid file format"}), 400

        # Save the uploaded file temporarily
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        try:
            # Predict the class
            predicted_class, confidence = predict_image(filepath)
            
            # Clean up the temporary file
            os.remove(filepath)

            return jsonify({
                "class": predicted_class,
                "confidence": confidence
            })

        except Exception as e:
            # Clean up the temporary file in case of error
            if os.path.exists(filepath):
                os.remove(filepath)
            raise e

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)