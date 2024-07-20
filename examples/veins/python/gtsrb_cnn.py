import numpy as np
import pandas as pd
import cv2
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
tf.keras.utils.set_random_seed(42)

num_classes = 43
input_shape = (48, 48, 3)
batch_size = 64
epochs = 5

df = pd.read_csv('GTSRB/raw/Train.csv')
df = df.sample(frac=1).reset_index(drop=True)

imgs = []
labels = []
for index, row in df.iterrows():
    img = cv2.imread('GTSRB/raw/' + row['Path'])

    min_side = min(img.shape[:-1])
    centre = img.shape[0]//2, img.shape[1]//2
    img = img[centre[0]-min_side//2:centre[0]+min_side//2,
              centre[1]-min_side//2:centre[1]+min_side//2, :]

    img = cv2.resize(img, (48, 48)) / 255.0
    imgs.append(img)
    labels.append(int(row['ClassId']))

x_train = np.array(imgs)
y_train = keras.utils.to_categorical(np.array(labels), num_classes)

model = keras.Sequential(
    [
        keras.Input(shape=input_shape),
        layers.Conv2D(1, kernel_size=(1, 1), padding="same", activation="relu"),
        layers.Conv2D(29, kernel_size=(5, 5), padding="same", activation="relu"),
        layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)),
        layers.Conv2D(59, kernel_size=(3, 3), padding="same", activation="relu"),
        layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)),
        layers.Conv2D(74, kernel_size=(3, 3), padding="same", activation="relu"),
        layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(300, activation="relu"),
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
