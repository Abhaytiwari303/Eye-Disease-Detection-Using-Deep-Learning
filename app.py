from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
import cv2

app = Flask(__name__)

# Load trained model
model = tf.keras.models.load_model("best_model.h5")

def process_image(image):
    img = cv2.imdecode(np.fromstring(image.read(), np.uint8), cv2.IMREAD_COLOR)
    img = cv2.resize(img, (224, 224))  # Resize to model's input shape
    img = img / 255.0  # Normalize
    img = np.expand_dims(img, axis=0)
    return img

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "GET":
        return render_template("predict.html")

    if "file" not in request.files:
        return jsonify({"result": "No file uploaded"})

    file = request.files["file"]
    img = process_image(file)
    
    prediction = model.predict(img)
    disease_class = np.argmax(prediction)

    diseases = ["normal", "Cataract", "Glaucoma", "Diabetic Retinopathy"]
    result = "It is " + diseases[disease_class]

    return jsonify({"result": result})

if __name__ == "__main__":
    app.run(debug=True)
