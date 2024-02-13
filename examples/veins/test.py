import os
import pickle
import random
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.cluster import AffinityPropagation
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
from python.src import constants, models, metrics


data = np.load(constants.DATA_PATH + 'v0_0_0_data.npz')
X, y = data['images_train'], data['labels_train']
num_classes = data['num_classes']

skf = StratifiedKFold(n_splits=5)
for i, (train_index, valid_index) in enumerate(skf.split(X, y)):
    if constants.SPLIT == i:
        X_train, y_train = X[train_index], keras.utils.to_categorical(y[train_index], num_classes)
        X_valid, y_valid = X[valid_index], keras.utils.to_categorical(y[valid_index], num_classes)

datagen = ImageDataGenerator(zoom_range=0.2, horizontal_flip=True)
datagen.fit(X_train)

model = models.get_model()
model.summary()
model.fit(datagen.flow(X_train, y_train, batch_size=50), steps_per_epoch = 3 * X_train.shape[0] / 50, epochs=50, validation_data=(X_valid, y_valid))
#model.fit(X_train, y_train, epochs=50, validation_data=(X_valid, y_valid))

'''
weights = []
for v in range(100):
    weights_path = 'python/experiments/CIFAR10_rot4/OurMethodRSURandomCossim3Epochs/weights/'
    with open(weights_path + 'v' + str(v) + '_weights.pickle', 'rb') as file:
        data = pickle.load(file)
        weights.append(data)

benchmark = random.randrange(100)
bw = metrics.flatten(weights[benchmark])
cossims = []
for w in weights:
    cossim = metrics.cossim(bw, metrics.flatten(w))
    cossims.append([cossim])
clustering = AffinityPropagation(damping=0.7, max_iter=2000).fit(cossims)
print(benchmark)
print(cossims)
print(clustering.labels_)
'''
