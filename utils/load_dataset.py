import os
import numpy as np
from utils.audio_features import extract_features

labels = {
    "hunger":0,
    "pain":1,
    "discomfort":2,
    "sleepiness":3
}

def load_dataset(dataset_path):

    X = []
    y = []

    for label in labels:

        folder = os.path.join(dataset_path, label)

        for file in os.listdir(folder):

            file_path = os.path.join(folder, file)

            features = extract_features(file_path)

            if features is not None:

                X.append(features)
                y.append(labels[label])

    return np.array(X), np.array(y)