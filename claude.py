import os
from flask import Flask, request, render_template, flash, redirect, url_for
from werkzeug.utils import secure_filename
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import *
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from collections import deque

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'

# Constants
IMAGE_HEIGHT, IMAGE_WIDTH = 64, 64
SEQUENCE_LENGTH = 16
CLASSES_LIST = ["NonViolence", "Violence"]

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Create upload folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def create_model():
    """Recreate the model architecture"""
    # Create MobileNetV2 base
    mobilenet = MobileNetV2(include_top=False, weights="imagenet")
    
    # Fine-tuning settings
    mobilenet.trainable = True
    for layer in mobilenet.layers[:-40]:
        layer.trainable = False
    
    # Create the model
    model = Sequential()
    
    model.add(Input(shape=(SEQUENCE_LENGTH, IMAGE_HEIGHT, IMAGE_WIDTH, 3)))
    model.add(TimeDistributed(mobilenet))
    model.add(Dropout(0.25))
    model.add(TimeDistributed(Flatten()))
    
    lstm_fw = LSTM(units=32)
    lstm_bw = LSTM(units=32, go_backwards=True)
    
    model.add(Bidirectional(lstm_fw, backward_layer=lstm_bw))
    model.add(Dropout(0.25))
    
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.25))
    
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.25))
    
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.25))
    
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.25))
    
    model.add(Dense(len(CLASSES_LIST), activation='softmax'))
    
    return model

# Load the model
model = None
try:
    model = create_model()
    model.load_weights('MoBiLSTM_model.h5')
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def predict_video(video_path):
    """Predict violence in a video file"""
    try:
        video_reader = cv2.VideoCapture(video_path)
        frames_list = []
        
        # Get total frames
        video_frames_count = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
        skip_frames_window = max(int(video_frames_count/SEQUENCE_LENGTH), 1)

        # Extract frames
        for frame_counter in range(SEQUENCE_LENGTH):
            video_reader.set(cv2.CAP_PROP_POS_FRAMES, frame_counter * skip_frames_window)
            success, frame = video_reader.read()
            
            if not success:
                break
                
            resized_frame = cv2.resize(frame, (IMAGE_HEIGHT, IMAGE_WIDTH))
            normalized_frame = resized_frame / 255
            frames_list.append(normalized_frame)

        video_reader.release()

        # Make prediction if we have enough frames
        if len(frames_list) == SEQUENCE_LENGTH:
            frames_array = np.expand_dims(frames_list, axis=0)
            predicted_labels_probabilities = model.predict(frames_array)[0]
            predicted_label = np.argmax(predicted_labels_probabilities)
            confidence = float(predicted_labels_probabilities[predicted_label])
            
            return {
                'prediction': CLASSES_LIST[predicted_label],
                'confidence': confidence,
                'success': True
            }
        else:
            return {
                'success': False,
                'error': 'Could not extract enough frames from video'
            }

    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'video' not in request.files:
        flash('No video file uploaded')
        return redirect(request.url)
    
    video_file = request.files['video']
    
    if video_file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    
    if not allowed_file(video_file.filename):
        flash('Invalid file type. Please upload mp4, avi, or mov files.')
        return redirect(request.url)
    
    if model is None:
        flash('Model not loaded. Please check server logs.')
        return redirect(request.url)
    
    try:
        # Save uploaded file
        filename = secure_filename(video_file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        video_file.save(filepath)
        
        # Make prediction
        result = predict_video(filepath)
        
        # Clean up
        os.remove(filepath)
        
        if result['success']:
            prediction = result['prediction']
            confidence = result['confidence']
            flash(f'Prediction: {prediction} (Confidence: {confidence:.2%})')
        else:
            flash(f'Error during prediction: {result.get("error", "Unknown error")}')
            
        return redirect(url_for('home'))
        
    except Exception as e:
        flash(f'Error processing video: {str(e)}')
        return redirect(url_for('home'))

if __name__ == '__main__':
    app.run(debug=True)