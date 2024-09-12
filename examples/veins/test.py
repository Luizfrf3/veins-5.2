import os
import pickle
import random
import numpy as np
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
from sklearn.cluster import AffinityPropagation
from sklearn.metrics import balanced_accuracy_score
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
from python.src import constants, models, metrics

class BalancedAccuracyCallback(keras.callbacks.Callback):

    def __init__(self, X_train, y_train, X_valid, y_valid):
        super(BalancedAccuracyCallback, self).__init__()
        self.X_train = X_train
        self.y_train = y_train
        self.X_valid = X_valid
        self.y_valid = y_valid

    def on_epoch_end(self, epoch, logs={}):
        y_pred = tf.argmax(self.model.predict(self.X_train), axis=1)
        y_true = tf.argmax(self.y_train, axis=1)
        train_balanced_accuracy = balanced_accuracy_score(y_true, y_pred)
        logs["train_balanced_accuracy"] = train_balanced_accuracy

        y_pred = tf.argmax(self.model.predict(self.X_valid), axis=1)
        y_true = tf.argmax(self.y_valid, axis=1)
        valid_balanced_accuracy = balanced_accuracy_score(y_true, y_pred)
        logs["valid_balanced_accuracy"] = valid_balanced_accuracy

data = np.load(constants.DATA_PATH + 'v0_data.npz')
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
history = model.fit(datagen.flow(X_train, y_train, batch_size=50), steps_per_epoch = 3 * X_train.shape[0] / 50, epochs=5, validation_data=(X_valid, y_valid), callbacks=[BalancedAccuracyCallback(X_train, y_train, X_valid, y_valid)])
print(history.history)
y_pred = tf.argmax(model.predict(X_valid), axis=1)
y_true = tf.argmax(y_valid, axis=1)
valid_balanced_accuracy = balanced_accuracy_score(y_true, y_pred)
print(valid_balanced_accuracy)
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

'''
weights = []
for v in ['v22', 'v91', 'v72', 'v66', 'v89', 'v87', 'v90', 'v95', 'v71', 'v9', 'v27']:
#for v in ['v43', 'v10', 'v27', 'v11', 'v34', 'v20', 'v41', 'v51', 'v37', 'v67', 'v66']:
    weights_path = 'python/experiments/CIFAR10_rot2/OurMethodRSURandomExtremaWeightedCossim/weights/'
    with open(weights_path + v + '_weights.pickle', 'rb') as file:
        data = pickle.load(file)
        weights.append(data)

bw = metrics.flatten(weights[0])
cossims = []
for w in weights:
    cossim = metrics.cossim(bw, metrics.flatten(w))
    cossims.append(cossim)
print(cossims)
'''
