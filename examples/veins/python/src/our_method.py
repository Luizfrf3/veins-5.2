import logging
import numpy as np
from tensorflow import keras
from sklearn.cluster import AffinityPropagation
from python.src import models, constants, logs

received_weights = {}

def _flatten(w):
    r = np.array([], dtype=np.float32)
    for i in range(len(w)):
        r = np.concatenate((r, w[i].flatten()), axis=0)
    return r

def _cossim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def _cka(features_x, features_y):
    features_x = features_x - np.mean(features_x, 0, keepdims=True)
    features_y = features_y - np.mean(features_y, 0, keepdims=True)
    dot_product_similarity = np.linalg.norm(features_x.T.dot(features_y)) ** 2
    normalization_x = np.linalg.norm(features_x.T.dot(features_x))
    normalization_y = np.linalg.norm(features_y.T.dot(features_y))
    return dot_product_similarity / (normalization_x * normalization_y)

def _cca(features_x, features_y):
    qx, _ = np.linalg.qr(features_x)
    qy, _ = np.linalg.qr(features_y)
    return np.linalg.norm(qx.T.dot(qy)) ** 2 / min(features_x.shape[1], features_y.shape[1])

def _local_clustering(vehicle_id, model, mw, X_valid, y_valid):
    loss, accuracy = model.evaluate(X_valid, y_valid, verbose=0)
    mfeatures = [1.0, 1.0, 1.0, loss, accuracy]
    inter_model = keras.Model(inputs=model.input, outputs=model.get_layer("final_dense").output)
    activation = np.array(inter_model(X_valid))

    rfeatures = []
    rweights = []
    for rweight in received_weights[vehicle_id].values():
        rmodel = models.get_model()
        rmodel.set_weights(rweight)

        loss, accuracy = rmodel.evaluate(X_valid, y_valid, verbose=0)
        cossim = _cossim(mw, _flatten(rweight))
        rinter_model = keras.Model(inputs=rmodel.input, outputs=rmodel.get_layer("final_dense").output)
        ractivation = np.array(rinter_model(X_valid))
        cka = _cka(activation, ractivation)
        cca = _cca(activation, ractivation)

        rfeatures.append([cossim, cka, cca, loss, accuracy])
        rweights.append(rweight)

    rfeatures = np.array(rfeatures)
    clustering = AffinityPropagation().fit(rfeatures)

    vehicle_cluster = clustering.predict([mfeatures])[0]
    indexes = [i for i in range(len(clustering.labels_)) if clustering.labels_[i] == vehicle_cluster]
    return [rweights[i] for i in indexes]

def store_weights(raw_weights, vehicle_id, sender_id):
    weights = models.decode_weights(raw_weights)
    if vehicle_id not in received_weights.keys():
        received_weights[vehicle_id] = {}
    received_weights[vehicle_id][sender_id] = weights

def train(vehicle_id, training_round, sim_time, vehicle_data, vehicle_models):
    X_train, y_train = vehicle_data[vehicle_id]['train']
    X_valid, y_valid = vehicle_data[vehicle_id]['valid']

    model = vehicle_models[vehicle_id]
    mweights = model.get_weights()

    if vehicle_id not in received_weights.keys():
        received_weights[vehicle_id] = {}

    if len(received_weights[vehicle_id]) > 0:
        clustered_weights = _local_clustering(vehicle_id, model, _flatten(mweights), X_valid, y_valid)

        for i in range(len(mweights)):
            for w in clustered_weights:
                mweights[i] = mweights[i] + w[i]
            mweights[i] = mweights[i] / (len(clustered_weights) + 1)
        model.set_weights(mweights)

    history = model.fit(X_train, y_train, epochs=constants.EPOCHS, validation_data=(X_valid, y_valid), verbose=0)

    logging.warning('Node {}, Training Round {}, History {}'.format(vehicle_id, training_round, history.history))
    logs_data = {'event': 'train', 'vehicle_id': vehicle_id, 'sim_time': sim_time, 'training_round': training_round, 'number_of_received_models': len(received_weights[vehicle_id]), 'history': history.history}
    logs.register_log(logs_data)

    models.save_weights(vehicle_id, model.get_weights())
    vehicle_models[vehicle_id] = model

    received_weights[vehicle_id] = {}
