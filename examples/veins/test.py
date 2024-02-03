import numpy as np
from sklearn.model_selection import StratifiedKFold
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
from python.src import constants, models

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
model.fit(datagen.flow(X_train, y_train, batch_size=50), steps_per_epoch = 3 * X_train.shape[0] / 50, epochs=50, validation_data=(X_valid, y_valid))
