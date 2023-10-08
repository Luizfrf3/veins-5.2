import os
import json
import numpy as np

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
    images_train = np.array(x_train[user], dtype=np.float32)
    labels_train = np.array(y_train[user], dtype=np.float32)
    images_test = np.array(x_test[user], dtype=np.float32)
    labels_test = np.array(y_test[user], dtype=np.float32)
    name = 'FEMNIST/data/v' + str(total) + '_' + user + '_data.npz'
    np.savez(
            name, images_train=images_train, labels_train=labels_train,
            images_test=images_test, labels_test=labels_test, num_classes=num_classes
        )
    total += 1
