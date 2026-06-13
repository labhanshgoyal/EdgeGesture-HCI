"""
Hand Tracker Module.
Wraps MediaPipe Hands to detect and track hand landmarks from video frames.
"""

import mediapipe as mp
import cv2
import numpy as np

class HandTracker:
    def __init__(self, max_hands=1, detection_conf=0.7, tracking_conf=0.5):
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_hands,
            min_detection_confidence=detection_conf,
            min_tracking_confidence=tracking_conf,
        )

    def process_frame(self, frame):
        """Process a video frame and detect hands.
        Args:
            frame: A BGR image (numpy array) from OpenCV.
        Returns:
            results: MediaPipe detection results.
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        return results

    def get_landmark_array(self, results):
        """
        Extract hand landmarks as a NumPy array.
        Args:
            results: The results from process_frame().
        Returns:
            numpy array of shape (21, 3) with [x, y, z] per landmark,
            or None if no hand detected.
        """
        if results.multi_hand_landmarks is None:
            return None
        hand_landmarks = results.multi_hand_landmarks[0]
        landmarks = []
        for lm in hand_landmarks.landmark:
            landmarks.append([lm.x, lm.y, lm.z])
        return np.array(landmarks)

    def draw_landmarks(self, frame, results):
        """
        Draw hand landmarks and connections on the frame.
        Args:
            frame: The BGR image to draw on.
            results: The results from process_frame().
        Returns:
            frame with landmarks drawn on it.
        """
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style(),
                )
        return frame

    def get_finger_states(self, landmarks_array):
        """
        Determine which fingers are extended (open) or folded (closed).
        Args:
            landmarks_array: NumPy array of shape (21, 3).
        Returns:
            List of 5 booleans: [thumb, index, middle, ring, pinky].
            True = extended, False = folded.
        """
        if landmarks_array is None:
            return None
        tips = [4, 8, 12, 16, 20]
        pip_joints = [3, 6, 10, 14, 18]
        finger_states = []
        # Thumb: compare x-position (thumb moves sideways)
        if landmarks_array[tips[0]][0] < landmarks_array[pip_joints[0]][0]:
            finger_states.append(True)
        else:
            finger_states.append(False)
        # Other 4 fingers: tip above (lower y) its joint = extended
        for i in range(1, 5):
            if landmarks_array[tips[i]][1] < landmarks_array[pip_joints[i]][1]:
                finger_states.append(True)
            else:
                finger_states.append(False)
        return finger_states

    def release(self):
        """Release MediaPipe resources."""
        self.hands.close()
