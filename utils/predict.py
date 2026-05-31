import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from config import MODEL_PATH, IMAGE_SIZE

# Load model once
model = load_model("model/weed_model.h5", compile=False)
print("Model loaded successfully")
def predict_image(img_path):

    # Load image
    img = image.load_img(img_path, target_size=IMAGE_SIZE)

    # Preprocess
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    prediction = model.predict(img_array)[0][0]

    # Label + confidence
    if prediction > 0.5:
        label = "🌿 Weed"
        confidence = prediction * 100
    else:
        label = "🌱 Crop"
        confidence = (1 - prediction) * 100

    return (label, round(confidence, 2))