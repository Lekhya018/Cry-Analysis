import librosa
import numpy as np

def extract_features(file_path):

    try:
        audio, sr = librosa.load(file_path, res_type='kaiser_fast')

        mfcc = np.mean(
            librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40).T,
            axis=0
        )

        zcr = np.mean(librosa.feature.zero_crossing_rate(audio))

        spectral_centroid = np.mean(
            librosa.feature.spectral_centroid(y=audio, sr=sr)
        )

        chroma = np.mean(
            librosa.feature.chroma_stft(y=audio, sr=sr)
        )

        features = np.hstack([mfcc, zcr, spectral_centroid, chroma])

        return features

    except Exception as e:
        print("Error processing file:", file_path)
        return None