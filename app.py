from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import numpy as np
import cv2
import os
import json

app = Flask(__name__)

# Load trained model
MODEL_PATH = "Alzheimer_model.h5"
model = tf.keras.models.load_model(MODEL_PATH)

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

# Function to preprocess image
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (244, 244))
    img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
    img = np.expand_dims(img, axis=0)
    return img

# Risk calculation function
def calculate_risk(age, family_history, conditions, lifestyle_factors):
    risk_score = 0
    impact_messages = []

    # Age-related risk
    if age >= 60:
        risk_score += 5
        impact_messages.append("🧓 Age Factor: Higher age is a known risk factor for Alzheimer's.")

    # Family history impact
    if family_history == "yes":
        risk_score += 2
        impact_messages.append("👪 Family History: A genetic predisposition increases the likelihood of Alzheimer's.")

    # Medical conditions impact
    condition_risk = {
        "diabetes": (1.5, "Diabetes affects blood flow to the brain, increasing risk."),
        "hypertension": (1.2, "High blood pressure can damage brain vessels, contributing to cognitive decline."),
        "cardiovascular": (1.8, "Cardiovascular issues reduce oxygen supply to the brain, increasing dementia risk."),
        "high_cholesterol": (1.3, "High cholesterol can lead to plaque buildup, affecting brain function.")
    }
    for condition in conditions:
        if condition in condition_risk:
            risk_value, reason = condition_risk[condition]
            risk_score += risk_value
            impact_messages.append(f" {condition.capitalize()}: {reason}")

    # Lifestyle impact
    lifestyle_risk = {
        "smoking": (1.8, "Smoking accelerates brain aging and increases oxidative stress."),
        "sedentary": (1.5, "A sedentary lifestyle reduces brain activity and increases the risk of neurodegeneration."),
        "poor_diet": (1.4, "A diet lacking essential nutrients impacts brain health and cognitive function."),
        "alcohol": (1.6, "Excessive alcohol consumption can lead to brain shrinkage and memory issues.")
    }
    for factor in lifestyle_factors:
        if factor in lifestyle_risk:
            risk_value, reason = lifestyle_risk[factor]
            risk_score += risk_value
            impact_messages.append(f" {factor.replace('_', ' ').capitalize()}: {reason}")

    return risk_score, "\n".join(impact_messages)


# Prediction API
@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file part"})

    file = request.files["file"]
    if file.filename == "" or not allowed_file(file.filename):
        return jsonify({"error": "Invalid file format"})

    # Save & preprocess image
    file_path = os.path.join("uploads", file.filename)
    file.save(file_path)
    image = preprocess_image(file_path)
    prediction = model.predict(image)

    # Decode class
    class_labels = ['Mild Demented', 'Moderate Demented', 'Non Demented', 'Very Mild Demented']
    predicted_class = class_labels[np.argmax(prediction)]
    confidence = float(np.max(prediction))

    # Get additional user inputs
    age = int(request.form["age"])
    family_history = request.form["family_history"]
    conditions = json.loads(request.form["conditions"])
    lifestyle_factors = json.loads(request.form["lifestyle_factors"])

    # Compute risk assessment
    risk_score, lifestyle_impact = calculate_risk(age, family_history, conditions, lifestyle_factors)

    os.remove(file_path)

    return jsonify({
        "prediction": predicted_class,
        "confidence": confidence,
        "risk_score": risk_score,
        "lifestyle_impact": lifestyle_impact
    })

@app.route("/")
def home():
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
