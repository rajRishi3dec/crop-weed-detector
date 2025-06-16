import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import os

# Load the trained model
model = load_model("model/weed_model.h5")

def predict_image(img_path):
    # Load and preprocess the image
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)[0][0]
    label = "Weed" if prediction > 0.5 else "Crop"
    confidence = prediction if prediction > 0.5 else 1 - prediction

    # Visualize result
    plt.imshow(img)
    plt.title(f"{label} ({confidence*100:.2f}%)")
    plt.axis('off')
    plt.show()

    return label
