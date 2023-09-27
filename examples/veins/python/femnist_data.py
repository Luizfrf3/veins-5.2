import os
import json
import numpy as np
from tensorflow import keras

num_classes = 62

x_train = {}
y_train = {}
x_test = {}
y_test = {}
for file_name in os.listdir('FEMNIST/raw'):
    with open('FEMNIST/raw/' + file_name, 'r') as f:
        data = json.load(f)
        if 'test' in file_name:
            for user in data['users']:
                x_test[user] = data['user_data'][user]['x']
                y_test[user] = data['user_data'][user]['y']
        else:
            for user in data['users']:
                x_train[user] = data['user_data'][user]['x']
                y_train[user] = data['user_data'][user]['y']

total = 0
for user in x_train.keys():
    train_validation_split = int(len(x_train[user]) * 0.8)
    images_train = np.array(x_train[user][:train_validation_split], dtype=np.float32)
    labels_train = keras.utils.to_categorical(np.array(y_train[user][:train_validation_split], dtype=np.float32), num_classes)
    images_validation = np.array(x_train[user][train_validation_split:], dtype=np.float32)
    labels_validation = keras.utils.to_categorical(np.array(y_train[user][train_validation_split:], dtype=np.float32), num_classes)
    images_test = np.array(x_test[user], dtype=np.float32)
    labels_test = keras.utils.to_categorical(np.array(y_test[user], dtype=np.float32), num_classes)
    name = 'FEMNIST/data/' + str(total) + '_' + user + '_data.npz'
    np.savez(
            name, images_train=images_train, labels_train=labels_train,
            images_validation=images_validation, labels_validation=labels_validation,
            images_test=images_test, labels_test=labels_test
        )
    total += 1
