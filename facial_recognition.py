import kagglehub
import os

# Download latest version of FER-2013
# to /root/.cache/kagglehub/datasets/msambare/fer2013/...
path = kagglehub.dataset_download("msambare/fer2013")

print("Path to dataset files:", path)
# List contents to verify structure (usually contains 'train' and 'test' folders)
print("Contents:", os.listdir(path))

# Update dataset_path to use the Kaggle download location
dataset_path = path
train_dir = os.path.join(dataset_path, 'train')
test_dir = os.path.join(dataset_path, 'test')

# Install necessary packages for hardware monitoring
!pip install pynvml

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import time
import psutil
import pynvml
import matplotlib.pyplot as plt

# Initialize NVML for GPU power tracking
pynvml.nvmlInit()
handle = pynvml.nvmlDeviceGetHandleByIndex(0) # Get the first GPU
gpu_name = pynvml.nvmlDeviceGetName(handle)
print(f"Configured to use GPU: {gpu_name}")

# --- Global Hyperparameters ---
BATCH_SIZE = 128
LEARNING_RATE = 0.002
EPOCHS = 100
PLATEAU_PATIENCE = 4
L1_FILTERS = 128
L2_FILTERS = 256
L3_FILTERS = 512
DENSE_UNITS = 256
DENSE_UNITS_2 = 128
DENSE_UNITS_3 = 64 # NEW: Parameterized 3rd Dense Layer

# --- Dropout Rates ---
DROP_1 = 0.02 # Layer 1 (Was 0.05)
DROP_2 = 0.05 # Layer 2 (Was 0.10)
DROP_3 = 0.10 # Layer 3 (Was 0.20)
DROP_4 = 0.10 # Dense Layer 1 (Was 0.30)
DROP_5 = 0.10 # Dense Layer 2 (Was 0.25)
DROP_6 = 0.10 # NEW: Dense Layer 3

# --- Data Preprocessing and Augmentation (tf.data.Dataset) ---

# Define the exact emotion classes based on the dataset folders
emotion_classes = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

# 1. Load the datasets natively using tf.keras.utils
train_ds = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(48, 48),
    batch_size=BATCH_SIZE,
    color_mode='grayscale',
    label_mode='categorical',
    class_names=emotion_classes
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(48, 48),
    batch_size=BATCH_SIZE,
    color_mode='grayscale',
    label_mode='categorical',
    class_names=emotion_classes
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    image_size=(48, 48),
    batch_size=BATCH_SIZE,
    color_mode='grayscale',
    label_mode='categorical',
    class_names=emotion_classes,
    shuffle=False
)

# 2. Define augmentations using Keras layers
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.02, fill_mode='nearest'), # Reduced from 0.055
    layers.RandomZoom(0.08, fill_mode='nearest'), # Reduced from 0.2
    layers.RandomTranslation(height_factor=0.1, width_factor=0.1, fill_mode='nearest'), # Reduced from 0.2 for both height and width
])

# 3. Apply augmentations and optimize pipeline with prefetching
AUTOTUNE = tf.data.AUTOTUNE

def prepare_train_ds(ds):
    ds = ds.map(lambda x, y: (data_augmentation(x / 255.0, training=True), y), num_parallel_calls=AUTOTUNE)
    return ds.prefetch(buffer_size=AUTOTUNE)

def prepare_eval_ds(ds):
    ds = ds.map(lambda x, y: (x / 255.0, y), num_parallel_calls=AUTOTUNE)
    return ds.prefetch(buffer_size=AUTOTUNE)

train_generator = prepare_train_ds(train_ds)
val_generator = prepare_eval_ds(val_ds)
test_generator = prepare_eval_ds(test_ds)

print("Optimized tf.data pipeline ready and fixed!")

from tensorflow.keras import regularizers

DROPOUT_RATE = 0.4      # Increase from 0.25 — your model is large enough to need this
L2_REG = 1e-4

# --- GPU Model Architecture ---
def build_gpu_model():
    model = models.Sequential()

    # --- Conv Block 1: 64 filters, double conv effect via larger kernel ---
    # 5x5 kernel on the first layer captures broader facial feature patterns
    # (eyes, nose width) that 3x3 misses at 48x48 resolution
    model.add(layers.Conv2D(64, (5, 5), padding='same', activation='elu',
                            input_shape=(48, 48, 1),
                            kernel_regularizer=regularizers.l2(L2_REG)))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))          # → 24x24
    model.add(layers.SpatialDropout2D(0.25))        # Better than Dropout after conv layers

    # --- Conv Block 2: 128 filters ---
    model.add(layers.Conv2D(128, (3, 3), padding='same', activation='elu',
                            kernel_regularizer=regularizers.l2(L2_REG)))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))          # → 12x12
    model.add(layers.SpatialDropout2D(0.25))

    # --- Conv Block 3: 256 filters (not 1024 — that was causing overfitting) ---
    model.add(layers.Conv2D(256, (3, 3), padding='same', activation='elu',
                            kernel_regularizer=regularizers.l2(L2_REG)))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))          # → 6x6
    model.add(layers.SpatialDropout2D(0.3))

    model.add(layers.Flatten())                     # 6x6x256 = 9,216 — manageable

    # Dense Layer 1
    model.add(layers.Dense(DENSE_UNITS, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(DROP_4))

    # Dense Layer 2
    model.add(layers.Dense(DENSE_UNITS_2, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(DROP_5))

    # Dense Layer 3
    model.add(layers.Dense(DENSE_UNITS_3, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(DROP_6))

    # --- Dense Block: Two layers instead of one for more representational power ---
    model.add(layers.Dense(512, activation='elu',
                           kernel_regularizer=regularizers.l2(L2_REG)))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(DROPOUT_RATE))

    model.add(layers.Dense(256, activation='elu',   # ✅ Second dense layer
                           kernel_regularizer=regularizers.l2(L2_REG)))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(DROPOUT_RATE))

    model.add(layers.Dense(7, activation='softmax'))
    return model

model = build_gpu_model()
model.summary()

# --- Compilation & Callbacks ---

# Adam optimizer handles noise/sparse gradients well, ideal for our dataset
opt = optimizers.Adam(learning_rate=LEARNING_RATE)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

# Set up training callbacks to improve efficiency and preserve the best weights
callbacks_list = [
    # Increased patience to 10 to prevent premature stopping if the model gets temporarily stuck
    callbacks.EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True),
    callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=PLATEAU_PATIENCE, min_lr=1e-6),
    callbacks.ModelCheckpoint('best_gpu_model.keras', monitor='val_accuracy', save_best_only=True)
]

# --- Training & Benchmarking ---
import numpy as np
from sklearn.utils.class_weight import compute_class_weight

# Check VRAM before starting
info = pynvml.nvmlDeviceGetMemoryInfo(handle)
print(f"Total VRAM: {info.total / 1024**2:.2f} MB")
print(f"Free VRAM: {info.free / 1024**2:.2f} MB")

print("Computing class weights to handle dataset imbalance...")
# Extract true labels to compute weights natively from the dataset
y_true = np.concatenate([np.argmax(y.numpy(), axis=-1) for x, y in train_ds])
class_weights = compute_class_weight('balanced', classes=np.unique(y_true), y=y_true)
class_weight_dict = dict(enumerate(class_weights))
print(f"Class Weights: {class_weight_dict}")

print("\nStarting training and power monitoring...")
start_time = time.time()

# Get baseline power usage
initial_power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0

# Execute training
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=val_generator,
    callbacks=callbacks_list,
    class_weight=class_weight_dict # Apply the computed weights here
)

end_time = time.time()
final_power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
total_time = end_time - start_time

print(f"\nTotal Training Time: {total_time:.2f} seconds")
print(f"Average Time per Epoch: {total_time / len(history.epoch):.2f} seconds")
print(f"Final GPU Power Usage: {final_power:.2f} W")
print(f"Approximate Energy Consumption: {((initial_power + final_power) / 2) * total_time:.2f} Joules")

# Evaluate the model on the test data
test_loss, test_accuracy = model.evaluate(test_generator)
print(f"\nTest Accuracy: {test_accuracy * 100:.2f}%")
print(f"Test Loss: {test_loss:.4f}")

# Visualize training & validation metrics
plt.figure(figsize=(12, 5))

# Accuracy subplot
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training vs Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

# Loss subplot
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training vs Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()