import os
import json
import numpy as np

num_classes = 62
num_clients = 100

x_train = []
y_train = []
x_test = []
y_test = []
for file_name in os.listdir('FEMNIST/raw'):
    with open('FEMNIST/raw/' + file_name, 'r') as f:
        data = json.load(f)
        if 'test' in file_name:
            for user in data['users']:
                x_test += data['user_data'][user]['x']
                y_test += data['user_data'][user]['y']
        else:
            for user in data['users']:
                x_train += data['user_data'][user]['x']
                y_train += data['user_data'][user]['y']

x_train = np.array_split(x_train, num_clients)
y_train = np.array_split(y_train, num_clients)
x_test = np.array_split(x_test, num_clients)
y_test = np.array_split(y_test, num_clients)

for i in range(len(x_train)):
    itr = np.array(x_train[i], dtype=np.float32)
    itr = itr.reshape((itr.shape[0], 28, 28))
    itr = 1.0 - itr
    images_train = np.expand_dims(itr, -1)
    labels_train = np.array(y_train[i], dtype=np.float32)
    its = np.array(x_test[i], dtype=np.float32)
    its = its.reshape((its.shape[0], 28, 28))
    its = 1.0 - its
    images_test = np.expand_dims(its, -1)
    labels_test = np.array(y_test[i], dtype=np.float32)
    name = 'FEMNIST/data/v' + str(i) + '_data.npz'
    np.savez(
        name, images_train=images_train, labels_train=labels_train,
        images_test=images_test, labels_test=labels_test, num_classes=num_classes
    )
