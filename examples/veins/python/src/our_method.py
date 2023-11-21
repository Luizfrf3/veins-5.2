import logging
import numpy as np
from tensorflow import keras
from sklearn.cluster import AffinityPropagation
from python.src import models, constants, logs, metrics

received_weights = {}
dataset_sizes = {}

def _local_clustering(node_id, model, mw, X_valid, y_valid):
    loss, accuracy = model.evaluate(X_valid, y_valid, verbose=0)
    #mfeatures = [1.0, 1.0, 1.0, loss, accuracy]
    mfeatures = [1.0, loss, accuracy]
    inter_model = keras.Model(inputs=model.input, outputs=model.get_layer("final_dense").output)
    activation = np.array(inter_model(X_valid))

    rfeatures = []
    rweights = []
    senders = []
    for sender, rweight in received_weights[node_id].items():
        rmodel = models.get_model()
        rmodel.set_weights(rweight)

        loss, accuracy = rmodel.evaluate(X_valid, y_valid, verbose=0)
        #cossim = metrics.cossim(mw, metrics.flatten(rweight))
        rinter_model = keras.Model(inputs=rmodel.input, outputs=rmodel.get_layer("final_dense").output)
        ractivation = np.array(rinter_model(X_valid))
        cka = metrics.cka(activation, ractivation)
        #cca = metrics.cca(activation, ractivation)

        #rfeatures.append([cossim, cka, cca, loss, accuracy])
        rfeatures.append([cka, loss, accuracy])
        rweights.append(rweight)
        senders.append(sender)

    rfeatures = np.array(rfeatures)
    clustering = AffinityPropagation().fit(rfeatures)

    vehicle_cluster = clustering.predict([mfeatures])[0]
    indexes = [i for i in range(len(clustering.labels_)) if clustering.labels_[i] == vehicle_cluster]
    return [{'w': rweights[i], 'id': senders[i]} for i in indexes]

def store_weights(raw_weights, dataset_size, node_id, sender_id):
    weights = models.decode_weights(raw_weights)
    if node_id not in received_weights.keys():
        received_weights[node_id] = {}
        dataset_sizes[node_id] = {}
    received_weights[node_id][sender_id] = weights
    dataset_sizes[node_id][sender_id] = dataset_size

def train(node_id, training_round, sim_time, vehicle_data, node_models):
    X_train, y_train = vehicle_data[node_id]['train']
    X_valid, y_valid = vehicle_data[node_id]['valid']

    model = node_models[node_id]
    mweights = model.get_weights()

    if node_id not in received_weights.keys():
        received_weights[node_id] = {}
        dataset_sizes[node_id] = {}

    if len(received_weights[node_id]) > 0:
        clustered_weights = _local_clustering(node_id, model, metrics.flatten(mweights), X_valid, y_valid)

        for i in range(len(mweights)):
            sizes = len(X_train)
            mweights[i] = mweights[i] * sizes
            for cw in clustered_weights:
                mweights[i] = mweights[i] + (cw['w'][i] * dataset_sizes[node_id][cw['id']])
                sizes += dataset_sizes[node_id][cw['id']]
            mweights[i] = mweights[i] / sizes
        model.set_weights(mweights)

    history = model.fit(X_train, y_train, epochs=constants.EPOCHS, validation_data=(X_valid, y_valid), verbose=0)

    logging.warning('Node {}, Training Round {}, History {}'.format(node_id, training_round, history.history))
    logs_data = {'event': 'train', 'node_id': node_id, 'sim_time': sim_time, 'training_round': training_round, 'number_of_received_models': len(received_weights[node_id]), 'history': history.history}
    logs.register_log(logs_data)

    models.save_weights(node_id, model.get_weights())
    node_models[node_id] = model

    received_weights[node_id] = {}
    dataset_sizes[node_id] = {}
