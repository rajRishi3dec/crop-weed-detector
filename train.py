from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
import os

# Constants
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 15
TRAIN_DIR = "dataset/train"
VAL_DIR = "dataset/val"
MODEL_PATH = "model/weed_model.h5"

# Create model directory if not exists
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

# Data preparation with augmentation for training
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest"
)

# Only rescale validation images
val_datagen = ImageDataGenerator(rescale=1./255)

# Load datasets
train_data = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary"
)

val_data = val_datagen.flow_from_directory(
    VAL_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary"
)

# 🔥 Show class mapping (Important!)
print(f"Class indices (Label Mapping): {train_data.class_indices}")

# Load MobileNetV2 as base model
base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224, 224, 3))

# Freeze base model layers if needed (optional)
base_model.trainable = False

x = GlobalAveragePooling2D()(base_model.output)
output = Dense(1, activation="sigmoid")(x)

model = Model(inputs=base_model.input, outputs=output)

# Compile and train
model.compile(optimizer=Adam(learning_rate=0.0001), loss="binary_crossentropy", metrics=["accuracy"])

# Summary
print(model.summary())

# Training
history = model.fit(
    train_data,
    epochs=EPOCHS,
    validation_data=val_data
)

# Save the model
model.save(MODEL_PATH)
print(f"✅ Model saved to {MODEL_PATH}")
