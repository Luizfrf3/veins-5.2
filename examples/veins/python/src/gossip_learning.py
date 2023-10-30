import logging
from python.src import constants, models, logs

def train(node_id, training_round, sim_time, vehicle_data, node_models):
    X_train, y_train = vehicle_data[node_id]['train']
    X_valid, y_valid = vehicle_data[node_id]['valid']

    model = node_models[node_id]
    history = model.fit(X_train, y_train, epochs=constants.EPOCHS, validation_data=(X_valid, y_valid), verbose=0)

    logging.warning('Node {}, Training Round {}, History {}'.format(node_id, training_round, history.history))
    logs_data = {'event': 'train', 'node_id': node_id, 'sim_time': sim_time, 'training_round': training_round, 'history': history.history}
    logs.register_log(logs_data)

    models.save_weights(node_id, model.get_weights())
    node_models[node_id] = model

def merge(raw_weights, dataset_size, node_id, vehicle_data, node_models):
    size = len(vehicle_data[node_id]['train'][0])
    model = node_models[node_id]
    received_weights = models.decode_weights(raw_weights)
    weights = model.get_weights()
    for i in range(len(weights)):
        weights[i] = (weights[i] * size + received_weights[i] * dataset_size) / (size + dataset_size)
    model.set_weights(weights)
