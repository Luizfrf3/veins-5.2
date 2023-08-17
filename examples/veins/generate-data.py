import tensorflow as tf
from tensorflow.keras import datasets
from tensorflow.keras.utils import to_categorical
import numpy as np

(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

images_dict = {}
labels_dict = {}
for i in range(10):
    images_dict['flow' + str(i % 2) + '.' + str(i % 5)] = train_images[5000 * i:5000 * (i + 1)]
    labels_dict['flow' + str(i % 2) + '.' + str(i % 5)] = train_labels[5000 * i:5000 * (i + 1)]
images_dict['train'], labels_dict['train'] = train_images, train_labels
images_dict['test'], labels_dict['test'] = test_images, test_labels

for k in images_dict.keys():
    images = np.array(images_dict[k], dtype=np.float32)
    labels = to_categorical(np.array(labels_dict[k], dtype=np.float32), 10)
    np.savez('data/' + k + '_data.npz', images=images, labels=labels)
