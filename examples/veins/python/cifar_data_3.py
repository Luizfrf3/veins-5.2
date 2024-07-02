import numpy as np
from tensorflow.keras import datasets
np.random.seed(42)

(x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()

x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255

indices_train = np.in1d(y_train, [1, 2, 3, 4, 5])
xtrain = np.copy(x_train[indices_train])
ytrain = np.copy(y_train[indices_train])
p = np.random.permutation(xtrain.shape[0])
xtrain = xtrain[p]
ytrain = ytrain[p]
indices_test = np.in1d(y_test, [1, 2, 3, 4, 5])
xtest = np.copy(x_test[indices_test])
ytest = np.copy(y_test[indices_test])
p = np.random.permutation(xtest.shape[0])
xtest = xtest[p]
ytest = ytest[p]

xtrvs = [xtrain[i::55] for i in range(40)]
ytrvs = [ytrain[i::55] for i in range(40)]
xtsvs = [xtest[i::55] for i in range(40)]
ytsvs = [ytest[i::55] for i in range(40)]

total = 0
for v in range(40):
    images_train = np.array(xtrvs[v], dtype=np.float32)
    labels_train = np.array(ytrvs[v], dtype=np.float32)
    images_test = np.array(xtsvs[v], dtype=np.float32)
    labels_test = np.array(ytsvs[v], dtype=np.float32)
    name = 'CIFAR10/data/v' + str(total) + '_0_' + str(v) + '_data.npz'
    np.savez(
        name, images_train=images_train, labels_train=labels_train,
        images_test=images_test, labels_test=labels_test, num_classes=10
    )
    total += 1

xtrvs = [xtrain[i::55] for i in range(40, 55)]
ytrvs = [ytrain[i::55] for i in range(40, 55)]
xtsvs = [xtest[i::55] for i in range(40, 55)]
ytsvs = [ytest[i::55] for i in range(40, 55)]

for v in range(15):
    images_train = np.array(xtrvs[v], dtype=np.float32)
    labels_train = np.array(ytrvs[v], dtype=np.float32)
    images_test = np.array(xtsvs[v], dtype=np.float32)
    labels_test = np.array(ytsvs[v], dtype=np.float32)
    name = 'CIFAR10/data/v' + str(total) + '_1_' + str(v) + '_data.npz'
    np.savez(
        name, images_train=images_train, labels_train=labels_train,
        images_test=images_test, labels_test=labels_test, num_classes=10
    )
    total += 1

indices_train = np.in1d(y_train, [0, 6, 7, 8, 9])
xtrain = np.copy(x_train[indices_train])
ytrain = np.copy(y_train[indices_train])
p = np.random.permutation(xtrain.shape[0])
xtrain = xtrain[p]
ytrain = ytrain[p]
indices_test = np.in1d(y_test, [0, 6, 7, 8, 9])
xtest = np.copy(x_test[indices_test])
ytest = np.copy(y_test[indices_test])
p = np.random.permutation(xtest.shape[0])
xtest = xtest[p]
ytest = ytest[p]

xtrvs = [xtrain[i::45] for i in range(30)]
ytrvs = [ytrain[i::45] for i in range(30)]
xtsvs = [xtest[i::45] for i in range(30)]
ytsvs = [ytest[i::45] for i in range(30)]

for v in range(30):
    images_train = np.array(xtrvs[v], dtype=np.float32)
    labels_train = np.array(ytrvs[v], dtype=np.float32)
    images_test = np.array(xtsvs[v], dtype=np.float32)
    labels_test = np.array(ytsvs[v], dtype=np.float32)
    name = 'CIFAR10/data/v' + str(total) + '_2_' + str(v) + '_data.npz'
    np.savez(
        name, images_train=images_train, labels_train=labels_train,
        images_test=images_test, labels_test=labels_test, num_classes=10
    )
    total += 1

xtrvs = [xtrain[i::45] for i in range(30, 45)]
ytrvs = [ytrain[i::45] for i in range(30, 45)]
xtsvs = [xtest[i::45] for i in range(30, 45)]
ytsvs = [ytest[i::45] for i in range(30, 45)]

for v in range(15):
    images_train = np.array(xtrvs[v], dtype=np.float32)
    labels_train = np.array(ytrvs[v], dtype=np.float32)
    images_test = np.array(xtsvs[v], dtype=np.float32)
    labels_test = np.array(ytsvs[v], dtype=np.float32)
    name = 'CIFAR10/data/v' + str(total) + '_3_' + str(v) + '_data.npz'
    np.savez(
        name, images_train=images_train, labels_train=labels_train,
        images_test=images_test, labels_test=labels_test, num_classes=10
    )
    total += 1
