import numpy as np
from tensorflow.keras import datasets
np.random.seed(42)

num_classes = 10
vehicles = 100
cluster_labels = [[0, 1, 3, 5, 7], [2, 4, 6, 8, 9]]
max_clusters = 2
vehicles_group_size = int(vehicles / max_clusters)

(x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()

x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255

cxtrain = []
cytrain = []
cxtest = []
cytest = []
for c in range(max_clusters):
    indices_train = np.in1d(y_train, cluster_labels[c])
    cxtrain.append(x_train[indices_train])
    cytrain.append(y_train[indices_train])

    indices_test = np.in1d(y_test, cluster_labels[c])
    cxtest.append(x_test[indices_test])
    cytest.append(y_test[indices_test])

total = 0
for c in range(max_clusters):
    p = np.random.permutation(cxtrain[c].shape[0])
    x_train_c = cxtrain[c][p]
    y_train_c = cytrain[c][p]
    p = np.random.permutation(cxtest[c].shape[0])
    x_test_c = cxtest[c][p]
    y_test_c = cytest[c][p]

    xtrvs = [x_train_c[i::vehicles_group_size] for i in range(vehicles_group_size)]
    ytrvs = [y_train_c[i::vehicles_group_size] for i in range(vehicles_group_size)]
    xtsvs = [x_test_c[i::vehicles_group_size] for i in range(vehicles_group_size)]
    ytsvs = [y_test_c[i::vehicles_group_size] for i in range(vehicles_group_size)]

    for v in range(vehicles_group_size):
        images_train = np.expand_dims(np.array(xtrvs[v], dtype=np.float32), -1)
        labels_train = np.array(ytrvs[v], dtype=np.float32)
        images_test = np.expand_dims(np.array(xtsvs[v], dtype=np.float32), -1)
        labels_test = np.array(ytsvs[v], dtype=np.float32)
        name = 'MNIST/data/v' + str(total) + '_' + str(c) + '_' + str(v) + '_data.npz'
        np.savez(
            name, images_train=images_train, labels_train=labels_train,
            images_test=images_test, labels_test=labels_test, num_classes=num_classes
        )
        total += 1
