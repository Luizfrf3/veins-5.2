import numpy as np
import json
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
tf.keras.utils.set_random_seed(42)

num_classes = 62
input_shape = (28, 28, 1)
batch_size = 64
epochs = 50

x_train = []
y_train = []
with open('FEMNIST/data.json') as f:
    data = json.load(f)
    for k,v in data["user_data"].items():
        x_train += v["x"]
        y_train += v["y"]
size = len(x_train)

x_train = np.reshape(np.array(x_train, dtype=np.float32), (size, 28, 28))
y_train = keras.utils.to_categorical(np.array(y_train, dtype=np.float32), num_classes)

model = keras.Sequential(
    [
        keras.Input(shape=input_shape),
        layers.Conv2D(32, kernel_size=(5, 5), padding="same", activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
        layers.Conv2D(64, kernel_size=(5, 5), padding="same", activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
        layers.Flatten(),
        layers.Dense(2048, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax")
    ]
)
model.summary()

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.2)

score = model.evaluate(x_train, y_train, verbose=0)
print("Train loss:", score[0])
print("Train accuracy:", score[1])
