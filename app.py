from flask import Flask, render_template, request
from utils.predict import predict_image
from config import UPLOAD_FOLDER
import os
import uuid
#import webbrowser
import threading 
import time
# Reduce TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    image_path = None

    if request.method == "POST":
        file = request.files.get("image")

        if file and file.filename != "":
            # Ensure static folder exists
            os.makedirs(UPLOAD_FOLDER, exist_ok=True)

            # Create unique filename to avoid overwrite
            filename = str(uuid.uuid4()) + "_" + file.filename
            filepath = os.path.join(UPLOAD_FOLDER, filename)

            # Save file
            file.save(filepath)

            # Predict
            result = predict_image(filepath)

            # Send image to frontend
            image_path = "/" + filepath

    return render_template("index.html", result=result, image_path=image_path)


def open_browser():
    time.sleep(1)  # wait for server to start
    # webbrowser.open("http://127.0.0.1:5000")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))

    # Open browser in separate thread
    threading.Thread(target=open_browser).start()

    app.run(host="0.0.0.0", port=port, debug=False)