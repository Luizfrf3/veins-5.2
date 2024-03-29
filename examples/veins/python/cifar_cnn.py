import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, datasets
tf.keras.utils.set_random_seed(42)

num_classes = 10
input_shape = (32, 32, 3)
batch_size = 64
epochs = 5

(x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()

x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = keras.Sequential(
    [
        keras.Input(shape=(32, 32, 3)),
        layers.Conv2D(6, kernel_size=(5, 5), padding="same", activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(16, kernel_size=(5, 5), padding="same", activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.25),
        layers.Flatten(),
        layers.Dense(120, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(84, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(10, name="final_dense", activation="softmax")
    ]
)
model.summary()

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.2)

score = model.evaluate(x_test, y_test, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])
