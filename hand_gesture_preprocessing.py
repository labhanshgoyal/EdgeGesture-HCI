# -*- coding: utf-8 -*-
"""
Hand Gesture Preprocessing — Robust Pipeline
Handles 18000+ samples with progress tracking, augmentation, and error handling.

Run on Google Colab (each section = 1 cell)
"""

# ═══════════════════════════════════════════════════════════════
# CELL 1: Setup + Mount Drive
# ═══════════════════════════════════════════════════════════════

from google.colab import drive
drive.mount('/content/drive')

import cv2
import numpy as np
import os
import time
import shutil

print("Setup done!")

# ═══════════════════════════════════════════════════════════════
# CELL 2: Unzip Dataset
# ═══════════════════════════════════════════════════════════════

import zipfile

ZIP_PATH = '/content/drive/MyDrive/gesture_subset.zip'
EXTRACT_PATH = '/content/dataset'

# Clean previous extraction if exists
if os.path.exists(EXTRACT_PATH):
    shutil.rmtree(EXTRACT_PATH)

print("Unzipping... (this may take a few minutes)")
with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
    zip_ref.extractall(EXTRACT_PATH)
print("Done!")

for item in os.listdir(EXTRACT_PATH):
    print(f"  {item}")

# ═══════════════════════════════════════════════════════════════
# CELL 3: Find gesture folders + count videos
# ═══════════════════════════════════════════════════════════════

DATASET_PATH = '/content/dataset/gesture_subset'  # Change if your path is different
FRAMES_DIR = '/content/frames'
SAVE_PATH = '/content/drive/MyDrive/gesture_data'
os.makedirs(SAVE_PATH, exist_ok=True)

FRAMES_PER_VIDEO = 30

gesture_classes = sorted([
    f for f in os.listdir(DATASET_PATH)
    if os.path.isdir(os.path.join(DATASET_PATH, f))
])

total_videos = 0
for gesture in gesture_classes:
    folder = os.path.join(DATASET_PATH, gesture)
    vids = [f for f in os.listdir(folder) if f.lower().endswith('.avi')]
    total_videos += len(vids)
    print(f"  {gesture}: {len(vids)} videos")

print(f"\nTotal: {total_videos} videos across {len(gesture_classes)} classes")
print(f"Expected frames: ~{total_videos * FRAMES_PER_VIDEO}")

# ═══════════════════════════════════════════════════════════════
# CELL 4: Extract frames from videos (with progress + error handling)
# ═══════════════════════════════════════════════════════════════

# Clear old frames
if os.path.exists(FRAMES_DIR):
    shutil.rmtree(FRAMES_DIR)

print("Extracting frames from all videos...")
start_time = time.time()
total_extracted = 0
failed_videos = []

for gesture in gesture_classes:
    video_folder = os.path.join(DATASET_PATH, gesture)
    frames_folder = os.path.join(FRAMES_DIR, gesture)
    os.makedirs(frames_folder, exist_ok=True)

    video_files = sorted([f for f in os.listdir(video_folder) if f.lower().endswith('.avi')])
    gesture_extracted = 0

    for video_file in video_files:
        video_path = os.path.join(video_folder, video_file)

        try:
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            if total_frames == 0:
                failed_videos.append(f"{gesture}/{video_file}: 0 frames")
                cap.release()
                continue

            num_to_extract = min(FRAMES_PER_VIDEO, total_frames)
            frame_indices = set([
                int(i * total_frames / num_to_extract)
                for i in range(num_to_extract)
            ])

            frame_count = 0
            saved_count = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_count in frame_indices:
                    video_name = video_file.replace('.avi', '')
                    frame_filename = f"{video_name}_f{saved_count:03d}.jpg"
                    save_path = os.path.join(frames_folder, frame_filename)
                    cv2.imwrite(save_path, frame)
                    saved_count += 1
                frame_count += 1

            cap.release()
            gesture_extracted += saved_count

        except Exception as e:
            failed_videos.append(f"{gesture}/{video_file}: {e}")
            continue

    total_extracted += gesture_extracted
    elapsed = time.time() - start_time
    print(f"  {gesture}: {gesture_extracted} frames  ({elapsed:.0f}s)")

print(f"\n{'='*50}")
print(f"Total frames extracted: {total_extracted}")
print(f"Time: {time.time() - start_time:.0f}s")

if failed_videos:
    print(f"\nFailed videos ({len(failed_videos)}):")
    for f in failed_videos[:10]:
        print(f"  {f}")

# ═══════════════════════════════════════════════════════════════
# CELL 5: Install MediaPipe + download model
# ═══════════════════════════════════════════════════════════════

import subprocess
subprocess.check_call(['pip', 'install', 'mediapipe'])

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import urllib.request

# Download hand landmarker model
MODEL_PATH = '/content/hand_landmarker.task'
if not os.path.exists(MODEL_PATH):
    MODEL_URL = 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task'
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Hand landmarker model downloaded!")

base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1
)
detector = vision.HandLandmarker.create_from_options(options)

print(f"MediaPipe {mp.__version__} ready!")

# ═══════════════════════════════════════════════════════════════
# CELL 6: Extract landmarks (batched, with progress)
# ═══════════════════════════════════════════════════════════════

all_landmarks = []
all_labels = []
skipped = 0
total_processed = 0

print("Extracting hand landmarks...")
start_time = time.time()

for gesture in gesture_classes:
    frames_folder = os.path.join(FRAMES_DIR, gesture)

    if not os.path.exists(frames_folder):
        print(f"  WARNING: {frames_folder} not found, skipping")
        continue

    frame_files = sorted([f for f in os.listdir(frames_folder) if f.endswith('.jpg')])
    detected = 0

    for frame_file in frame_files:
        frame_path = os.path.join(frames_folder, frame_file)

        try:
            mp_image = mp.Image.create_from_file(frame_path)
            result = detector.detect(mp_image)

            if result.hand_landmarks:
                landmarks = result.hand_landmarks[0]
                coords = []
                for lm in landmarks:
                    coords.append(lm.x)
                    coords.append(lm.y)
                    coords.append(lm.z)

                all_landmarks.append(coords)
                all_labels.append(gesture)
                detected += 1
            else:
                skipped += 1

        except Exception as e:
            skipped += 1
            continue

        total_processed += 1

        # Progress every 500 frames
        if total_processed % 500 == 0:
            elapsed = time.time() - start_time
            rate = total_processed / elapsed
            print(f"    ... {total_processed} processed ({rate:.0f} frames/sec)")

    detection_rate = (detected / len(frame_files) * 100) if frame_files else 0
    print(f"  {gesture}: {detected}/{len(frame_files)} detected ({detection_rate:.0f}%)")

detector.close()

X = np.array(all_landmarks)
y = np.array(all_labels)

elapsed = time.time() - start_time
print(f"\n{'='*50}")
print(f"Total samples:  {X.shape[0]}")
print(f"Skipped:        {skipped}")
print(f"X shape:        {X.shape}")
print(f"y shape:        {y.shape}")
print(f"Time:           {elapsed:.0f}s")

# ═══════════════════════════════════════════════════════════════
# CELL 7: Save raw data + Train/Val/Test split
# ═══════════════════════════════════════════════════════════════

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle

# Save raw data
np.save(os.path.join(SAVE_PATH, 'X_landmarks.npy'), X)
np.save(os.path.join(SAVE_PATH, 'y_labels.npy'), y)

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

print("Label mapping:")
for i, label in enumerate(label_encoder.classes_):
    count = np.sum(y_encoded == i)
    print(f"  {i:2d} = {label:12s} ({count} samples)")

# Split: 70/15/15
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y_encoded, test_size=0.30, random_state=42, stratify=y_encoded
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
)

print(f"\nTrain: {X_train.shape[0]} samples")
print(f"Val:   {X_val.shape[0]} samples")
print(f"Test:  {X_test.shape[0]} samples")

# Save splits
for name, data in [('X_train', X_train), ('y_train', y_train),
                    ('X_val', X_val), ('y_val', y_val),
                    ('X_test', X_test), ('y_test', y_test)]:
    np.save(os.path.join(SAVE_PATH, f'{name}.npy'), data)

with open(os.path.join(SAVE_PATH, 'label_encoder.pkl'), 'wb') as f:
    pickle.dump(label_encoder, f)

print("Split and saved!")

# ═══════════════════════════════════════════════════════════════
# CELL 8: Data Augmentation (makes model robust)
# ═══════════════════════════════════════════════════════════════

print(f"Original training samples: {X_train.shape[0]}")

augmented_X = [X_train.copy()]
augmented_y = [y_train.copy()]

# 1. Jitter — simulates microgravity hand trembling (4 levels)
for jitter in [0.005, 0.01, 0.02, 0.03]:
    X_jittered = X_train + np.random.normal(0, jitter, X_train.shape)
    # Only clip x,y columns — NOT z (z can be negative)
    for i in range(21):
        X_jittered[:, i*3]   = np.clip(X_jittered[:, i*3],   0, 1)  # x
        X_jittered[:, i*3+1] = np.clip(X_jittered[:, i*3+1], 0, 1)  # y
    augmented_X.append(X_jittered)
    augmented_y.append(y_train.copy())

# 2. Scale — simulates different camera distances (4 levels)
for scale in [0.9, 0.95, 1.05, 1.1]:
    X_scaled = X_train * scale
    for i in range(21):
        X_scaled[:, i*3]   = np.clip(X_scaled[:, i*3],   0, 1)
        X_scaled[:, i*3+1] = np.clip(X_scaled[:, i*3+1], 0, 1)
    augmented_X.append(X_scaled)
    augmented_y.append(y_train.copy())

# 3. Occlusion — simulates gloves hiding fingertips (3 levels)
fingertip_indices = [4, 8, 12, 16, 20]
for num_hide in [1, 2, 3]:
    X_occluded = X_train.copy()
    for j in range(len(X_occluded)):
        tips = np.random.choice(fingertip_indices, num_hide, replace=False)
        for tip in tips:
            X_occluded[j, tip*3]   = 0
            X_occluded[j, tip*3+1] = 0
            X_occluded[j, tip*3+2] = 0
    augmented_X.append(X_occluded)
    augmented_y.append(y_train.copy())

# Combine and shuffle
X_train_aug = np.concatenate(augmented_X, axis=0)
y_train_aug = np.concatenate(augmented_y, axis=0)

shuffle_idx = np.random.permutation(len(X_train_aug))
X_train_aug = X_train_aug[shuffle_idx]
y_train_aug = y_train_aug[shuffle_idx]

multiplier = X_train_aug.shape[0] / X_train.shape[0]
print(f"Augmented training samples: {X_train_aug.shape[0]} ({multiplier:.0f}x original)")
print(f"  + 4 jitter levels")
print(f"  + 4 scale levels")
print(f"  + 3 occlusion levels")

# ═══════════════════════════════════════════════════════════════
# CELL 9: Build + Train Robust Model
# ═══════════════════════════════════════════════════════════════

import tensorflow as tf
from tensorflow.keras import layers, models

NUM_CLASSES = len(label_encoder.classes_)

model = models.Sequential([
    layers.Input(shape=(63,)),
    layers.Reshape((21, 3)),

    # CNN layers — learn spatial patterns between landmarks
    layers.Conv1D(64, kernel_size=3, activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.Conv1D(128, kernel_size=3, activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.Conv1D(256, kernel_size=3, activation='relu', padding='same'),
    layers.BatchNormalization(),

    # GRU — learn sequential relationships
    layers.GRU(128, return_sequences=False),

    # Dense classification head
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),

    layers.Dense(NUM_CLASSES, activation='softmax'),
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

print(f"\nTraining on {X_train_aug.shape[0]} augmented samples...")

history = model.fit(
    X_train_aug, y_train_aug,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=64,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=7),
    ]
)

# ═══════════════════════════════════════════════════════════════
# CELL 10: Evaluate + Confusion Matrix + Save
# ═══════════════════════════════════════════════════════════════

from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Test accuracy
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"\n🎯 Test Accuracy: {test_acc * 100:.2f}%")

# Predictions
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

# Classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred_classes,
      target_names=label_encoder.classes_))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred_classes)
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_)
plt.title(f'Confusion Matrix — Robust Model ({test_acc*100:.1f}% Accuracy)')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.savefig(os.path.join(SAVE_PATH, 'confusion_matrix_robust.png'), dpi=150)
plt.show()

# Training curves
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
ax1.plot(history.history['accuracy'], label='Train')
ax1.plot(history.history['val_accuracy'], label='Validation')
ax1.set_title('Accuracy')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.plot(history.history['loss'], label='Train')
ax2.plot(history.history['val_loss'], label='Validation')
ax2.set_title('Loss')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(SAVE_PATH, 'training_curves_robust.png'), dpi=150)
plt.show()

# Save model
model.save(os.path.join(SAVE_PATH, 'gesture_model_robust.h5'))
print(f"\n✅ Robust model saved to: {SAVE_PATH}/gesture_model_robust.h5")
print(f"📊 Confusion matrix saved to: {SAVE_PATH}/confusion_matrix_robust.png")
print(f"📈 Training curves saved to: {SAVE_PATH}/training_curves_robust.png")