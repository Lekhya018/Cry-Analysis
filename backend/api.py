from fastapi import FastAPI, UploadFile, File
import tensorflow as tf
import numpy as np
import librosa

app = FastAPI()

# load trained model
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
async def predict(file: UploadFile = File(...)):

    # load audio
    audio, sr = librosa.load(file.file, sr=22050)

    # extract MFCC features
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    features = np.mean(mfcc.T, axis=0).reshape(1, -1)

    prediction = model.predict(features)

    result = classes[np.argmax(prediction)]

    return {"prediction": result}