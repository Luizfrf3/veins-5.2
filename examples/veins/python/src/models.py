import pickle
from tensorflow import keras
from tensorflow.keras import layers
from python.src import constants

def get_model(num_classes):
    model = None

    if constants.DATASET == constants.MNIST:
        model = keras.Sequential(
            [
                keras.Input(shape=(28, 28, 1)),
                layers.Conv2D(32, kernel_size=(3, 3), padding="same", activation="relu"),
                layers.MaxPooling2D(pool_size=(2, 2)),
                layers.Conv2D(64, kernel_size=(3, 3), padding="same", activation="relu"),
                layers.MaxPooling2D(pool_size=(2, 2)),
                layers.Dropout(0.25),
                layers.Flatten(),
                layers.Dense(128, activation="relu"),
                layers.Dropout(0.5),
                layers.Dense(num_classes, activation="softmax")
            ]
        )
    elif constants.DATASET == constants.CIFAR10:
        model = keras.Sequential(
            [
                keras.Input(shape=(32, 32, 3)),
                layers.Conv2D(32, kernel_size=(3, 3), padding="same", activation="relu"),
                layers.Conv2D(32, kernel_size=(3, 3), padding="same", activation="relu"),
                layers.MaxPooling2D(pool_size=(2, 2)),
                layers.Conv2D(64, kernel_size=(3, 3), padding="same", activation="relu"),
                layers.Conv2D(64, kernel_size=(3, 3), padding="same", activation="relu"),
                layers.MaxPooling2D(pool_size=(2, 2)),
                layers.Conv2D(128, kernel_size=(3, 3), padding="same", activation="relu"),
                layers.Conv2D(128, kernel_size=(3, 3), padding="same", activation="relu"),
                layers.MaxPooling2D(pool_size=(2, 2)),
                layers.Dropout(0.25),
                layers.Flatten(),
                layers.Dense(128, activation="relu"),
                layers.Dropout(0.5),
                layers.Dense(num_classes, activation="softmax")
            ]
        )
    elif constants.DATASET == constants.FEMNIST:
        model = keras.Sequential(
            [
                keras.Input(shape=(28, 28, 1)),
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
    elif constants.DATASET == constants.GTSRB:
        model = keras.Sequential(
            [
                keras.Input(shape=(48, 48, 3)),
                layers.Conv2D(1, kernel_size=(1, 1), padding="same", activation="relu"),
                layers.Conv2D(29, kernel_size=(5, 5), padding="same", activation="relu"),
                layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)),
                layers.Conv2D(59, kernel_size=(3, 3), padding="same", activation="relu"),
                layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)),
                layers.Conv2D(74, kernel_size=(3, 3), padding="same", activation="relu"),
                layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)),
                layers.Flatten(),
                layers.Dense(300, activation="relu"),
                layers.Dropout(0.5),
                layers.Dense(num_classes, activation="softmax")
            ]
        )

    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model

def save_weights(car_id, weights):
    weights_path = constants.WEIGHTS_FOLDER + car_id + constants.WEIGHTS_FILE_SUFFIX
    with open(weights_path, 'wb') as weights_file:
        pickle.dump(weights, weights_file)

def encode_weights(weights):
    weights_bytes = pickle.dumps(weights)
    raw_weights = ''
    for byte in weights_bytes:
        raw_weights += str(byte) + ','
    return raw_weights[:-1]

def decode_weights(raw_weights):
    byte_list = []
    for byte_str in raw_weights.split(','):
        byte_list.append(int(byte_str))
    weights_bytes = bytes(byte_list)
    return pickle.loads(weights_bytes)
