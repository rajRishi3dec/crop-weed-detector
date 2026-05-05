import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from config import MODEL_PATH, IMAGE_SIZE

model = load_model("model/weed_model.h5")
model.save("model/weed_model_new.keras")

def predict_image(img_path):
    img = image.load_img(img_path, target_size=IMAGE_SIZE)
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)[0][0]

    label = "🌿 Weed" if prediction > 0.5 else "🌱 Crop"
    confidence = prediction if prediction > 0.5 else 1 - prediction

    return f"{label} ({confidence*100:.2f}%)"