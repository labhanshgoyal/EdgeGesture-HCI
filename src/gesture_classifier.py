import numpy as np 
import os
import pickle
from collections import deque

class GestureClassifier:
    def __init__(self, model_path=None, encoder_path=None, buffer_size=15, movement_threshold=0.05):
        self.buffer_size=buffer_size
        self.movement_threshold=movement_threshold
        self.position_buffer=deque(maxlen=buffer_size)
        self.last_gesture="background"
        self.gesture_cooldown=0

        self.model=None
        self.label_encoder=None
        self.use_model=False

        if model_path and os.path.exists(model_path):
            try:
                import tensorflow as tf
                self.model = tf.keras.models.load_model(model_path)
                print(f"[Classifier] Model loaded: {model_path}")
                self.use_model=True
            except Exception as e:
                print(f"[Classifier] Failed to load model: {e}")
        
        if encoder_path and os.path.exists(encoder_path):
            with open(encoder_path, 'rb') as f:
                self.label_encoder=pickle.load(f)

    def update(self, landmarks_array): #Add new frame's landmarks to buffer
        if landmarks_array is None:
            self.position_buffer.clear()
            return

        wrist = landmarks_array[0]
        middle_tip = landmarks_array[12]

        self.position_buffer.append({
            "wrist": wrist.copy(),
            "middle_tip": middle_tip.copy(),
        })
    
    def classify(self, landmarks_array=None): # Classify the gesture. Uses model if available, else rule-based.
        if self.use_model and landmarks_array is not None:
            return self._classify_with_model(landmarks_array)
        else:
            return self._classify_rule_based()