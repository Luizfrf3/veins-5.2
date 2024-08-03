import logging
from tensorflow.python import keras
from keras.preprocessing.image import ImageDataGenerator
from python.src import constants, models, logs

clean_time = [constants.CLEAR_TIME]

def train(node_id, training_round, sim_time, vehicle_data, node_models):
    X_train, y_train = vehicle_data[node_id]['train']
    X_valid, y_valid = vehicle_data[node_id]['valid']

    model = node_models[node_id]
    if constants.DATA_AUGMENTATION:
        datagen = ImageDataGenerator(zoom_range=0.2, horizontal_flip=True)
        #datagen = ImageDataGenerator(zoom_range=0.2, rotation_range=15, width_shift_range=0.2, height_shift_range=0.2)
        datagen.fit(X_train)
        history = model.fit(datagen.flow(X_train, y_train, batch_size=constants.BATCH_SIZE), steps_per_epoch = constants.EPOCHS * X_train.shape[0] / 50, validation_data=(X_valid, y_valid), callbacks=[models.BalancedAccuracyCallback(X_train, y_train, X_valid, y_valid)], verbose=0)
    else:
        history = model.fit(X_train, y_train, epochs=constants.EPOCHS, batch_size=constants.BATCH_SIZE, validation_data=(X_valid, y_valid), callbacks=[models.BalancedAccuracyCallback(X_train, y_train, X_valid, y_valid)], verbose=0)

    logging.warning('Node {}, Training Round {}, History {}'.format(node_id, training_round, history.history))
    logs_data = {'event': 'train', 'node_id': node_id, 'sim_time': sim_time, 'training_round': training_round, 'history': history.history}
    logs.register_log(logs_data)

    models.save_weights(node_id, model.get_weights())
    node_models[node_id] = model

    if sim_time >= clean_time[0]:
        logging.warning('Clearing Keras session')
        models.clear_session()
        clean_time[0] = clean_time[0] + constants.CLEAR_TIME

def merge(raw_weights, dataset_size, node_id, sender_id, vehicle_data, node_models):
    size = len(vehicle_data[node_id]['train'][0])
    model = node_models[node_id]
    received_weights = models.decode_weights(raw_weights, sender_id)
    weights = model.get_weights()
    for i in range(len(weights)):
        weights[i] = (weights[i] * size + received_weights[i] * dataset_size) / (size + dataset_size)
    model.set_weights(weights)
    node_models[node_id] = model
