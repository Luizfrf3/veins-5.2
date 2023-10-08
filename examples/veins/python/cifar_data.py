import numpy as np
import cv2
from tensorflow.keras import datasets
np.random.seed(42)

num_classes = 10
vehicles = 12
clusters = [None, cv2.ROTATE_180, cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE]
max_clusters = 4
vehicles_group_size = int(vehicles / max_clusters)

(x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()

x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255

total = 0
for c in range(max_clusters):
    p = np.random.permutation(x_train.shape[0])
    x_train_c = x_train[p]
    y_train_c = y_train[p]
    p = np.random.permutation(x_test.shape[0])
    x_test_c = x_test[p]
    y_test_c = y_test[p]

    if clusters[c] is not None:
        x_train_c = [cv2.rotate(img, clusters[c]) for img in x_train_c]
        x_test_c = [cv2.rotate(img, clusters[c]) for img in x_test_c]

    xtrvs = [x_train_c[i::vehicles_group_size] for i in range(vehicles_group_size)]
    ytrvs = [y_train_c[i::vehicles_group_size] for i in range(vehicles_group_size)]
    xtsvs = [x_test_c[i::vehicles_group_size] for i in range(vehicles_group_size)]
    ytsvs = [y_test_c[i::vehicles_group_size] for i in range(vehicles_group_size)]

    for v in range(vehicles_group_size):
        images_train = np.array(xtrvs[v], dtype=np.float32)
        labels_train = np.array(ytrvs[v], dtype=np.float32)
        images_test = np.array(xtsvs[v], dtype=np.float32)
        labels_test = np.array(ytsvs[v], dtype=np.float32)
        name = 'CIFAR10/data/v' + str(total) + '_' + str(c) + '_' + str(v) + '_data.npz'
        np.savez(
            name, images_train=images_train, labels_train=labels_train,
            images_test=images_test, labels_test=labels_test, num_classes=num_classes
        )
        total += 1
