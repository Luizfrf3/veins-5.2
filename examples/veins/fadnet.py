import os
import pickle
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

WEIGHTS_FOLDER = 'weights/'
WEIGHTS_FILE_SUFFIX = '_weights.pickle'
DATA_FOLDER = 'data/'
DATA_FILE_SUFFIX = '_data.npz'
TEST_PATH = 'data/test_data.npz'
EPOCHS = 1

def _get_model():
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
    return model

def _encode_weights(weights):
    weights_bytes = pickle.dumps(weights)
    raw_weights = ''
    for byte in weights_bytes:
        raw_weights += str(byte) + ','
    return raw_weights[:-1]

def _decode_weights(raw_weights):
    byte_list = []
    for byte_str in raw_weights.split(','):
        byte_list.append(int(byte_str))
    weights_bytes = bytes(byte_list)
    return pickle.loads(weights_bytes)

def _save_weights(weights_path, weights):
    with open(weights_path, 'wb') as weights_file:
        pickle.dump(weights, weights_file)

def _read_weights(weights_path):
    weights = None
    if os.path.exists(weights_path):
        with open(weights_path, 'rb') as weights_file:
            weights = pickle.load(weights_file)
    else:
        weights = _get_model().get_weights()
        _save_weights(weights_path, weights)
    return weights

def get_weights(carId):
    weights_path = WEIGHTS_FOLDER + carId + WEIGHTS_FILE_SUFFIX
    weights = _read_weights(weights_path)
    return _encode_weights(weights)

def train(carId):
    tf.keras.utils.set_random_seed(42)

    train_path = DATA_FOLDER + carId + DATA_FILE_SUFFIX
    train_data = np.load(train_path)
    test_data = np.load(TEST_PATH)

    train_images, train_labels = train_data['images'], train_data['labels']
    test_images, test_labels = test_data['images'], test_data['labels']

    weights_path = WEIGHTS_FOLDER + carId + WEIGHTS_FILE_SUFFIX
    weights = _read_weights(weights_path)

    model = _get_model()
    model.set_weights(weights)

    model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
    history = model.fit(train_images, train_labels, epochs=EPOCHS, validation_data=(test_images, test_labels))
    print(carId, history.history)

    _save_weights(weights_path, model.get_weights())

def merge(raw_weights, carId):
    weights_path = WEIGHTS_FOLDER + carId + WEIGHTS_FILE_SUFFIX
    weights = _read_weights(weights_path)

    received_weights = _decode_weights(raw_weights)

    for i in range(len(weights)):
        weights[i] = (weights[i] + received_weights[i]) / 2
    
    _save_weights(weights_path, weights)
