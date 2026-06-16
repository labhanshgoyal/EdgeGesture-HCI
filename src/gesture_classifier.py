import numpy as np
from collections import deque

class GestureClassifier: #Tracks hand movement over multiple frames and classifies the direction of movement into one of 11 gesture classes.
    def __init__(self, buffer_size=15, movement_threshold=0.05):
        """Initialize the classifier.
        Args:
            buffer_size: Number of past frames to store for tracking movement.
            movement_threshold: Minimum Euclidean distance to move before recognizing a direction.
        """
        self.buffer_size = buffer_size
        self.movement_threshold = movement_threshold
        
        #deque = auto-removing oldest items when full (sliding window)
        self.position_buffer = deque(maxlen=buffer_size) 
        self.last_gesture = "background"
        self.gesture_cooldown = 0

    def update(self, landmarks_array):
        """
        Add new frame's landmarks to the tracking buffer.
        Args:
            landmarks_array: NumPy array of shape (21, 3) from hand_tracker.
        """
