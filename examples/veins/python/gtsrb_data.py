import numpy as np
import pandas as pd
import cv2
from tensorflow import keras
from sklearn.model_selection import train_test_split
np.random.seed(42)

num_classes = 43
num_clients = 10
alpha = 0.5

def read_data(path):
    df = pd.read_csv(path)
    df = df.sample(frac=1).reset_index(drop=True)

    imgs = []
    labels = []
    for index, row in df.iterrows():
        img = cv2.imread('GTSRB/raw/' + row['Path'])

        min_side = min(img.shape[:-1])
        centre = img.shape[0]//2, img.shape[1]//2
        img = img[centre[0]-min_side//2:centre[0]+min_side//2,
                centre[1]-min_side//2:centre[1]+min_side//2, :]

        img = cv2.resize(img, (48, 48)) / 255.0
        imgs.append(img)
        labels.append(int(row['ClassId']))

    imgs = np.array(imgs).astype("float32")
    labels = np.array(labels)

    p = np.random.permutation(imgs.shape[0])
    imgs = imgs[p]
    labels = labels[p]
    return imgs, labels

x_train, y_train = read_data('GTSRB/raw/Train.csv')
x_test, y_test = read_data('GTSRB/raw/Test.csv')

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
        proportions = np.array([p * (len(idx_j)< N / num_clients) for p, idx_j in zip(proportions, idx_batch)])
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
    images_train = np.array(x_train, dtype=np.float32)
    labels_train = keras.utils.to_categorical(np.array(y_train, dtype=np.float32), num_classes)
    images_test = np.array(x_test, dtype=np.float32)
    labels_test = keras.utils.to_categorical(np.array(y_test, dtype=np.float32), num_classes)
    name = 'GTSRB/data/' + str(i) + '_data.npz'
    np.savez(
            name, images_train=images_train, labels_train=labels_train,
            images_test=images_test, labels_test=labels_test
        )
