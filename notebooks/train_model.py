from utils.load_dataset import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import numpy as np

print("Loading dataset...")

X, y = load_dataset("dataset")

print("Dataset loaded")
print("Samples:", X.shape)

# normalize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training samples:", X_train.shape)

# build model
model = tf.keras.Sequential([

    tf.keras.layers.Dense(128, activation='relu'),

    tf.keras.layers.Dropout(0.3),

    tf.keras.layers.Dense(64, activation='relu'),

    tf.keras.layers.Dense(4, activation='softmax')

])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("Training model...")

model.fit(
    X_train,
    y_train,
    epochs=30,
    batch_size=16,
    validation_data=(X_test, y_test)
)

print("Evaluating model...")

loss, accuracy = model.evaluate(X_test, y_test)

print("Test Accuracy:", accuracy)

model.save("models/cry_model.h5")

print("Model saved to models/cry_model.h5")