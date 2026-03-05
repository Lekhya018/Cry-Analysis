import librosa
import numpy as np

def extract_features(file_path):

    # load audio file
    audio, sample_rate = librosa.load(file_path, duration=3)

    # extract MFCC features
    mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)

    # take mean of features
    features = np.mean(mfcc.T, axis=0)

    return features