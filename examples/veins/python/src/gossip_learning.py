import logging
from python.src import constants, models, logs

def train(vehicle_id, training_round, sim_time, vehicle_data, vehicle_models):
    X_train, y_train = vehicle_data[vehicle_id]['train']
    X_valid, y_valid = vehicle_data[vehicle_id]['valid']

    model = vehicle_models[vehicle_id]
    history = model.fit(X_train, y_train, epochs=constants.EPOCHS, validation_data=(X_valid, y_valid), verbose=0)

    logging.warning('Node {}, Training Round {}, History {}'.format(vehicle_id, training_round, history.history))
    logs_data = {'event': 'train', 'vehicle_id': vehicle_id, 'sim_time': sim_time, 'training_round': training_round, 'history': history.history}
    logs.register_log(logs_data)

    models.save_weights(vehicle_id, model.get_weights())
    vehicle_models[vehicle_id] = model

def merge(raw_weights, vehicle_id, vehicle_models):
    model = vehicle_models[vehicle_id]
    received_weights = models.decode_weights(raw_weights)
    weights = model.get_weights()
    for i in range(len(weights)):
        weights[i] = (weights[i] + received_weights[i]) / 2
    model.set_weights(weights)
