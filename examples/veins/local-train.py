import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
tf.keras.utils.set_random_seed(42)

EPOCHS = 30
TRAIN_PATH = 'data/train_data.npz'
TEST_PATH = 'data/test_data.npz'

train_data = np.load(TRAIN_PATH)
test_data = np.load(TEST_PATH)

train_images, train_labels = train_data['images'], train_data['labels']
test_images, test_labels = test_data['images'], test_data['labels']

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Dropout(0.2))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dropout(0.2))
model.add(layers.Dense(10))
model.summary()
print(type(model.get_weights()))

model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

history = model.fit(train_images, train_labels, epochs=EPOCHS, validation_data=(test_images, test_labels))
print(history.history)

train_loss, train_acc = model.evaluate(train_images,  train_labels, verbose=2)
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
print(train_loss, train_acc)
print(test_loss, test_acc)