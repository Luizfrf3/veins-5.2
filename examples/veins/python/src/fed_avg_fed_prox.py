import logging
import numpy as np
from tensorflow.python import keras
from keras.preprocessing.image import ImageDataGenerator
from python.src import constants, models, logs

received_weights = {}
dataset_sizes = {}

clean_time = [50]

def store_weights(raw_weights, dataset_size, node_id, sender_id):
    weights = models.decode_weights(raw_weights)
    if node_id not in received_weights.keys():
        received_weights[node_id] = {}
        dataset_sizes[node_id] = {}
    received_weights[node_id][sender_id] = weights
    dataset_sizes[node_id][sender_id] = dataset_size

def aggregation(aggregation_round, node_id, sim_time, node_models):
    model = node_models[node_id]

    if node_id not in received_weights.keys():
        received_weights[node_id] = {}
        dataset_sizes[node_id] = {}

    if len(received_weights[node_id]) > 0:
        weights = []
        model_weights = model.get_weights()
        for i in range(len(model_weights)):
            weights.append(np.zeros(model_weights[i].shape))

        for i in range(len(weights)):
            size = 0
            for sender, rweights in received_weights[node_id].items():
                weights[i] += rweights[i] * dataset_sizes[node_id][sender]
                size += dataset_sizes[node_id][sender]
            weights[i] = weights[i] / size
        model.set_weights(weights)

    logs_data = {'event': 'aggregation', 'node_id': node_id, 'sim_time': sim_time, 'aggregation_round': aggregation_round, 'number_of_received_models': len(received_weights[node_id])}
    logs.register_log(logs_data)

    models.save_weights(node_id, model.get_weights())
    node_models[node_id] = model

    received_weights[node_id] = {}
    dataset_sizes[node_id] = {}

def receive_global_model(raw_weights, node_id, sender_id, sim_time, node_models):
    logs_data = {'event': 'receive_global_model', 'node_id': node_id, 'sim_time': sim_time, 'sender_id': sender_id}
    logs.register_log(logs_data)

    model = node_models[node_id]
    weights = models.decode_weights(raw_weights)
    model.set_weights(weights)

def train(node_id, training_round, sim_time, vehicle_data, node_models):
    X_train, y_train = vehicle_data[node_id]['train']
    X_valid, y_valid = vehicle_data[node_id]['valid']

    model = node_models[node_id]
    if constants.DATA_AUGMENTATION:
        datagen = ImageDataGenerator(zoom_range=0.2, horizontal_flip=True)
        datagen.fit(X_train)
        history = model.fit(datagen.flow(X_train, y_train, batch_size=constants.BATCH_SIZE), steps_per_epoch = constants.EPOCHS * X_train.shape[0] / 50, validation_data=(X_valid, y_valid), verbose=0)
    else:
        history = model.fit(X_train, y_train, epochs=constants.EPOCHS, batch_size=constants.BATCH_SIZE, validation_data=(X_valid, y_valid), verbose=0)

    logging.warning('Node {}, Training Round {}, History {}'.format(node_id, training_round, history.history))
    logs_data = {'event': 'train', 'node_id': node_id, 'sim_time': sim_time, 'training_round': training_round, 'history': history.history}
    logs.register_log(logs_data)

    models.save_weights(node_id, model.get_weights())
    node_models[node_id] = model

    if sim_time >= clean_time[0]:
        logging.warning('Clearing Keras session')
        models.clear_session()
        clean_time[0] = clean_time[0] + 50
