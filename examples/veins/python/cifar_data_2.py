import numpy as np
from tensorflow.keras import datasets
from sklearn.model_selection import train_test_split
np.random.seed(42)

num_classes = 10
num_clients = 100
alpha = 0.5

(x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()

x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255

dataset_imgs = np.concatenate((x_train, x_test), axis=0)
dataset_labels = np.concatenate((y_train, y_test), axis=0)

X = [[] for _ in range(num_clients)]
y = [[] for _ in range(num_clients)]
dataidx_map = {}

least_samples = 32

min_size = 0
K = num_classes
N = len(dataset_labels)

try_cnt = 1
while min_size < least_samples:
    if try_cnt > 1:
        print(f'Client data size does not meet the minimum requirement {least_samples}. Try allocating again for the {try_cnt}-th time.')
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
    try_cnt += 1

for j in range(num_clients):
    dataidx_map[j] = idx_batch[j]

for client in range(num_clients):
    idxs = dataidx_map[client]
    X[client] = dataset_imgs[idxs]
    y[client] = dataset_labels[idxs]

for i in range(len(y)):
    x_train, x_test, y_train, y_test = train_test_split(X[i], y[i], train_size=0.8, shuffle=True)
    images_train = np.array(x_train, dtype=np.float32)
    labels_train = np.array(y_train, dtype=np.float32)
    images_test = np.array(x_test, dtype=np.float32)
    labels_test = np.array(y_test, dtype=np.float32)
    name = 'CIFAR10/data/v' + str(i) + '_data.npz'
    np.savez(
        name, images_train=images_train, labels_train=labels_train,
        images_test=images_test, labels_test=labels_test, num_classes=num_classes
    )
