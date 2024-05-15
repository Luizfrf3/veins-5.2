import numpy as np
from tensorflow.keras import datasets
from sklearn.model_selection import train_test_split
np.random.seed(42)

num_classes = 10
num_clients = 100
alpha = 0.1

(x_train, y_train), (x_test, y_test) = datasets.fashion_mnist.load_data()

x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255

dataset_imgs = np.concatenate((x_train, x_test), axis=0)
dataset_labels = np.concatenate((y_train, y_test), axis=0)

X = [[] for _ in range(num_clients)]
y = [[] for _ in range(num_clients)]
dataidx_map = {}

min_size = 0
K = num_classes
N = len(dataset_labels)
while min_size < num_classes:
    idx_batch = [[] for _ in range(num_clients)]
    for k in range(K):
        idx_k = np.where(dataset_labels == k)[0]
        np.random.shuffle(idx_k)
        proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
        proportions = np.array([p * (len(idx_j) < N / num_clients) for p, idx_j in zip(proportions, idx_batch)])
        proportions = proportions / proportions.sum()
        proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
        idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
        min_size = min([len(idx_j) for idx_j in idx_batch])

for j in range(num_clients):
    dataidx_map[j] = idx_batch[j]

for client in range(num_clients):
    idxs = dataidx_map[client]
    X[client] = dataset_imgs[idxs]
    y[client] = dataset_labels[idxs]

for i in range(len(y)):
    x_train, x_test, y_train, y_test = train_test_split(X[i], y[i], train_size=0.8, shuffle=True)
    images_train = np.expand_dims(np.array(x_train, dtype=np.float32), -1)
    labels_train = np.array(y_train, dtype=np.float32)
    images_test = np.expand_dims(np.array(x_test, dtype=np.float32), -1)
    labels_test = np.array(y_test, dtype=np.float32)
    name = 'FMNIST/data/v' + str(i) + '_data.npz'
    np.savez(
        name, images_train=images_train, labels_train=labels_train,
        images_test=images_test, labels_test=labels_test, num_classes=num_classes
    )
