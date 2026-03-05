from fastapi import FastAPI
import tensorflow as tf
import numpy as np

app = FastAPI()

model = tf.keras.models.load_model("models/cry_model.h5")

classes = [
    "belly pain",
    "burping",
    "cold_hot",
    "discomfort",
    "hungry",
    "lonely",
    "scared",
    "tired"
]

@app.get("/")
def home():
    return {"message": "Infant Cry Analyzer API running"}

@app.post("/predict")
def predict(features: list):

    features = np.array(features).reshape(1,-1)

    prediction = model.predict(features)

    result = classes[np.argmax(prediction)]

    return {"prediction": result}