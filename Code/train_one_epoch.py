import os, sys, time
import numpy as np
import pandas as pd
from PIL import Image

# Ensure working directory is Code directory
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

print("="*70)
print(" GDI PREDICTION MODEL - 1 EPOCH DRY RUN ON SYNTHETIC DENSEPOSE DATA")
print("="*70)

# Set seeds
np.random.seed(1)

import keras
from keras import layers, models, regularizers

# 1. Load Data
print("\n[Step 1/5] Loading metadata and dataset...")
csv_path = os.path.join(script_dir, 'data', 'imagepath_gdi_10.csv')
data_full = pd.read_csv(csv_path)

t0 = time.time()
video_clips = []
labels = []

# Sliding window extraction (10 frames per clip, stride 5)
input_window_size = 10
sample_stride = 5

for start in range(0, len(data_full) - input_window_size, sample_stride):
    end = start + input_window_size
    # Check patient ID match
    start_p = data_full['image_path'].iloc[start].split('/')[3]
    end_p = data_full['image_path'].iloc[end-1].split('/')[3]
    if start_p == end_p:
        frames = []
        for p in data_full['image_path'].iloc[start:end]:
            full_p = os.path.join(script_dir, p.lstrip('./'))
            img = Image.open(full_p)
            frames.append(np.array(img))
        video_clips.append(np.array(frames))
        labels.append(data_full['labels'].iloc[start])

video_clips = np.array(video_clips, dtype=np.float32) / 255.0
labels = np.array(labels, dtype=np.float32)

t_load = time.time() - t0
print(f"Loaded {len(video_clips)} video sequences ({len(video_clips)*10} frames total) in {t_load:.2f}s")
print(f"Video input tensor shape: {video_clips.shape}")
print(f"Target labels tensor shape: {labels.shape}")

# 2. Train/Validation Split
print("\n[Step 2/5] Creating train/validation split (80/20)...")
msk = np.random.rand(len(video_clips)) < 0.8
train_x, val_x = video_clips[msk], video_clips[~msk]
train_y, val_y = labels[msk], labels[~msk]
print(f"Train set: {len(train_x)} samples | Validation set: {len(val_x)} samples")

# 3. Build Best-Performer Architecture (ANet 2D CNN + LSTM)
print("\n[Step 3/5] Building spatiotemporal deep neural network (ANet CNN + LSTM)...")
conv_dim = 8
kernel_size = 8
dropout_amount = 0.5
reg = regularizers.l2(10**(-3.5))

frame_model = models.Sequential([
    layers.Input(shape=(480, 640, 3)),
    layers.Conv2D(conv_dim, kernel_size=kernel_size, padding='same'),
    layers.Activation('relu'),
    layers.BatchNormalization(),
    layers.Conv2D(conv_dim, kernel_size=kernel_size, padding='same'),
    layers.Activation('relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(pool_size=2),
    layers.Dropout(dropout_amount),
    layers.Conv2D(conv_dim, kernel_size=kernel_size, padding='same', kernel_regularizer=reg),
    layers.Activation('relu'),
    layers.BatchNormalization(),
    layers.Conv2D(conv_dim, kernel_size=kernel_size, padding='same', kernel_regularizer=reg),
    layers.Activation('relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(pool_size=2),
    layers.Dropout(dropout_amount),
    layers.Conv2D(conv_dim, kernel_size=kernel_size, padding='same', kernel_regularizer=reg),
    layers.Activation('relu'),
    layers.BatchNormalization(),
    layers.Conv2D(conv_dim, kernel_size=kernel_size, padding='same', kernel_regularizer=reg),
    layers.Activation('relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(pool_size=3),
    layers.Dropout(dropout_amount),
    layers.Flatten(),
    layers.Dense(16, activation='relu')
], name="ANet_Spatial_CNN")

video_input = layers.Input(shape=(10, 480, 640, 3), name="video_input_10_frames")
encoded_seq = layers.TimeDistributed(frame_model, name="TimeDistributed_ANet")(video_input)
encoded_video = layers.LSTM(256, name="Temporal_LSTM")(encoded_seq)
output = layers.Dense(1, activation='linear', name="GDI_Output")(encoded_video)

model = models.Model(inputs=video_input, outputs=output, name="GDI_Video_Model")
opt = keras.optimizers.Adam(learning_rate=0.001)
model.compile(loss='mse', metrics=['mae'], optimizer=opt)

print("Model Architecture Overview:")
model.summary()

# 4. Train 1 Epoch
print("\n[Step 4/5] Training for 1 epoch...")
t_train_start = time.time()
history = model.fit(
    train_x, train_y,
    validation_data=(val_x, val_y),
    batch_size=8,
    epochs=1,
    verbose=1
)
t_train_end = time.time()
epoch_time = t_train_end - t_train_start

# 5. Evaluate and Report
print("\n[Step 5/5] Validation & Results Summary")
print("="*70)
val_loss = history.history['val_loss'][0]
val_mae = history.history['val_mae'][0]
val_rmse = np.sqrt(val_loss)

train_loss = history.history['loss'][0]
train_mae = history.history['mae'][0]

print(f"Epoch 1/1 Completed in: {epoch_time:.2f} seconds ({epoch_time/len(train_x)*8:.2f}s per batch of 8)")
print(f"Training Loss (MSE):    {train_loss:.4f}")
print(f"Training MAE:           {train_mae:.4f}")
print(f"Validation Loss (MSE):  {val_loss:.4f}")
print(f"Validation MAE:         {val_mae:.4f}")
print(f"Validation RMSE:        {val_rmse:.4f}")

# Sample Predictions
preds = model.predict(val_x[:5])
print("\nSample Validation Predictions:")
print(f"{'Sample #':<10} | {'True GDI':<12} | {'Predicted GDI':<15} | {'Absolute Error':<15}")
print("-" * 60)
for i in range(len(preds)):
    diff = abs(val_y[i] - preds[i][0])
    print(f"Sample {i+1:<3} | {val_y[i]:<12.2f} | {preds[i][0]:<15.2f} | {diff:<15.2f}")

print("\n>>> EXECUTION SUCCESSFUL: Pipeline ran end-to-end with 0 errors!")
print("="*70)
