import logging
import copy
import math
import numpy as np
from tensorflow.python import keras
from keras.preprocessing.image import ImageDataGenerator
from sklearn.cluster import AffinityPropagation
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.signal import argrelextrema
from python.src import models, constants, logs, metrics

received_weights = {}
received_weights_while_training = {}
dataset_sizes = {}
dataset_sizes_while_training = {}

clean_time = [constants.CLEAR_TIME]

rmodel = models.get_model()

def _preprocess_activations(act):
    result = np.array(act)
    if len(act.shape) > 2:
        result = result.reshape((act.shape[0], act.shape[1] * act.shape[2] * act.shape[3]))

    #scaler = StandardScaler()
    #scaler.fit(result)
    #result = scaler.transform(result)

    #pca = PCA(n_components=0.99)
    #pca.fit(result)
    #result = pca.transform(result)

    return result

def _weighted_aggregation(node_id, model, mw, X_accept, y_accept):
    #loss, accuracy = model.evaluate(X_accept, y_accept, verbose=0)
    #inter_model = keras.Model(inputs=model.input, outputs=models.get_outputs(model))
    #activations = [_preprocess_activations(act) for act in inter_model(X_accept)]
    #mfeatures = [1.0 for _ in range(len(activations))]
    mfeatures = [1.0]
    #mfeatures = [1.0, 1.0, 1.0, loss, accuracy]
    #mfeatures = [metrics.cca(act, act, act.shape[1]) for act in activations]

    result = []
    for sender, rweight in received_weights[node_id].items():
        #loss, accuracy = rmodel.evaluate(X_accept, y_accept, verbose=0)
        cossim = metrics.cossim(mw, metrics.flatten(rweight))
        #rmodel.set_weights(rweight)
        #rinter_model = keras.Model(inputs=rmodel.input, outputs=models.get_outputs(rmodel))
        #ractivations = [_preprocess_activations(ract) for ract in rinter_model(X_accept)]
        #ckas = [metrics.cka(activations[i], ractivations[i]) for i in range(len(ractivations))]
        #ccas = [metrics.cca(activations[i], ractivations[i], min(activations[i].shape[1], ractivations[i].shape[1])) for i in range(len(ractivations))]

        result.append({
            'w': rweight,
            'f': (cossim + 1.0) / 2,
            'id': sender
        })

    #result.sort(key=lambda r: r['f'], reverse=True)
    #values = result[:math.ceil(len(result) / 4)]
    #return values, mfeatures[0]

    result.sort(key=lambda r: r['f'], reverse=False)
    diff = [result[i]['f'] - result[i - 1]['f'] for i in range(1, len(result))]
    rel_diff = [diff[i] / result[i]['f'] for i in range(len(diff))]
    arg = argrelextrema(np.array(rel_diff), np.greater)[0]
    values = result[arg[-1] + 1:] if len(arg) > 0 else result
    #return values, mfeatures[0]

    x = [values[i]['f'] for i in range(len(values))] + [1.0]
    e_x = np.exp(x - np.max(x))
    softmax = e_x / e_x.sum()
    for i in range(len(values)):
        values[i]['f'] = softmax[i]
    return values, softmax[-1]

def _local_clustering(node_id, model, mw, X_accept, y_accept):
    #loss, accuracy = model.evaluate(X_accept, y_accept, verbose=0)
    #inter_model = keras.Model(inputs=model.input, outputs=models.get_outputs(model))
    #activations = [_preprocess_activations(act) for act in inter_model(X_accept)]
    #mfeatures = [1.0 for _ in range(len(activations))]
    mfeatures = [1.0]
    #mfeatures = [1.0, 1.0, 1.0, loss, accuracy]
    #mfeatures = [metrics.cca(act, act, act.shape[1]) for act in activations]

    rfeatures = []
    rweights = []
    senders = []
    for sender, rweight in received_weights[node_id].items():
        #rmodel.set_weights(rweight)

        #loss, accuracy = rmodel.evaluate(X_accept, y_accept, verbose=0)
        cossim = metrics.cossim(mw, metrics.flatten(rweight))
        #rinter_model = keras.Model(inputs=rmodel.input, outputs=models.get_outputs(rmodel))
        #ractivations = [_preprocess_activations(ract) for ract in rinter_model(X_accept)]
        #ckas = [metrics.cka(activations[i], ractivations[i]) for i in range(len(ractivations))]
        #ccas = [metrics.cca(activations[i], ractivations[i], min(activations[i].shape[1], ractivations[i].shape[1])) for i in range(len(ractivations))]

        #rfeatures.append(ckas)
        #rfeatures.append([sum(ckas) / len(ckas) if np.isfinite(sum(ckas) / len(ckas)) else 0.0])
        rfeatures.append([cossim])
        rweights.append(rweight)
        senders.append(sender)

    #rfeatures.append(mfeatures)
    rfeatures = np.array(rfeatures)
    clustering = AffinityPropagation(damping=0.7, max_iter=2000).fit(rfeatures)

    vehicle_cluster = clustering.predict([mfeatures])[0]
    indexes = [i for i in range(len(clustering.labels_)) if clustering.labels_[i] == vehicle_cluster]
    #indexes = [i for i in range(len(clustering.labels_) - 1) if clustering.labels_[i] == clustering.labels_[-1]]
    return [{'w': rweights[i], 'id': senders[i]} for i in indexes]

def store_weights(raw_weights, dataset_size, node_id, sender_id):
    weights = models.decode_weights(raw_weights, sender_id)
    if node_id not in received_weights.keys():
        received_weights[node_id] = {}
        dataset_sizes[node_id] = {}
    #nid = int(node_id[1:])
    #sid = int(sender_id[1:])
    #if (nid < 30 and sid >= 30) or (nid < 60 and sid >= 60) or (nid < 90 and sid >= 90):
    #    return
    received_weights[node_id][sender_id] = weights
    dataset_sizes[node_id][sender_id] = dataset_size

def store_weights_while_training(raw_weights, dataset_size, node_id, sender_id):
    weights = models.decode_weights(raw_weights, sender_id)
    if node_id not in received_weights_while_training.keys():
        received_weights_while_training[node_id] = {}
        dataset_sizes_while_training[node_id] = {}
    #nid = int(node_id[1:])
    #sid = int(sender_id[1:])
    #if (nid < 30 and sid >= 30) or (nid < 60 and sid >= 60) or (nid < 90 and sid >= 90):
    #    return
    received_weights_while_training[node_id][sender_id] = weights
    dataset_sizes_while_training[node_id][sender_id] = dataset_size

def train(node_id, training_round, sim_time, vehicle_data, node_models):
    X_train, y_train = vehicle_data[node_id]['train']
    X_accept, y_accept = vehicle_data[node_id]['accept']
    X_valid, y_valid = vehicle_data[node_id]['valid']

    accepted_model = False
    maccuracy = 0.0
    raccuracy = 0.0
    model = node_models[node_id]
    mweights = model.get_weights()

    if node_id not in received_weights.keys():
        received_weights[node_id] = {}
        dataset_sizes[node_id] = {}
    if node_id not in received_weights_while_training.keys():
        received_weights_while_training[node_id] = {}
        dataset_sizes_while_training[node_id] = {}

    participating_nodes = []
    cluster_nodes = []

    if len(received_weights[node_id]) > 0:
        participating_nodes = [node for node in received_weights[node_id].keys()]

        clustered_weights = _local_clustering(node_id, model, metrics.flatten(mweights), X_accept, y_accept)
        cluster_nodes = [cw['id'] for cw in clustered_weights]
        for i in range(len(mweights)):
            sizes = len(X_train)
            mweights[i] = mweights[i] * sizes
            for cw in clustered_weights:
                mweights[i] = mweights[i] + (cw['w'][i] * dataset_sizes[node_id][cw['id']])
                sizes += dataset_sizes[node_id][cw['id']]
            mweights[i] = mweights[i] / sizes

        #clustered_weights, mf = _weighted_aggregation(node_id, model, metrics.flatten(mweights), X_accept, y_accept)
        #cluster_nodes = [cw['id'] for cw in clustered_weights]
        #for i in range(len(mweights)):
        #    wagg = len(X_train) * mf
        #    mweights[i] = mweights[i] * wagg
        #    for cw in clustered_weights:
        #        mweights[i] = mweights[i] + (cw['w'][i] * dataset_sizes[node_id][cw['id']] * cw['f'])
        #        wagg += dataset_sizes[node_id][cw['id']] * cw['f']
        #    mweights[i] = mweights[i] / wagg

        #for i in range(len(mweights)):
        #    size = len(X_train)
        #    mweights[i] = mweights[i] * size
        #    for sender, rweights in received_weights[node_id].items():
        #        mweights[i] += rweights[i] * dataset_sizes[node_id][sender]
        #        size += dataset_sizes[node_id][sender]
        #    mweights[i] = mweights[i] / size

        rmodel.set_weights(mweights)
        maccuracy, raccuracy = metrics.balanced_accuracy(model, rmodel, X_accept, y_accept)
        if raccuracy >= maccuracy or abs(maccuracy - raccuracy) <= constants.THRESHOLD:
            model.set_weights(mweights)
            accepted_model = True
        #model.set_weights(mweights)

    if constants.DATA_AUGMENTATION:
        datagen = ImageDataGenerator(zoom_range=0.1, rotation_range=15, width_shift_range=0.1, height_shift_range=0.1)
        datagen.fit(X_train)
        history = model.fit(datagen.flow(X_train, y_train, batch_size=constants.BATCH_SIZE), steps_per_epoch = constants.EPOCHS * X_train.shape[0] / 50, validation_data=(X_valid, y_valid), callbacks=[models.BalancedAccuracyCallback(X_train, y_train, X_valid, y_valid)], verbose=0)
    else:
        history = model.fit(X_train, y_train, epochs=constants.EPOCHS, batch_size=constants.BATCH_SIZE, validation_data=(X_valid, y_valid), callbacks=[models.BalancedAccuracyCallback(X_train, y_train, X_valid, y_valid)], verbose=0)

    logging.warning('Node {}, Training Round {}, History {}'.format(node_id, training_round, history.history))
    logs_data = {'event': 'train', 'node_id': node_id, 'sim_time': sim_time, 'accepted_model': accepted_model, 'maccuracy': maccuracy, 'raccuracy': raccuracy, 'training_round': training_round, 'number_of_received_models': len(received_weights[node_id]), 'history': history.history, 'participating_nodes': participating_nodes, 'cluster_nodes': cluster_nodes}
    logs.register_log(logs_data)

    models.save_weights(node_id, model.get_weights())
    node_models[node_id] = model

    received_weights[node_id] = copy.deepcopy(received_weights_while_training[node_id])
    dataset_sizes[node_id] = copy.deepcopy(dataset_sizes_while_training[node_id])
    received_weights_while_training[node_id] = {}
    dataset_sizes_while_training[node_id] = {}

    if sim_time >= clean_time[0]:
        logging.warning('Clearing Keras session')
        models.clear_session()
        clean_time[0] = clean_time[0] + constants.CLEAR_TIME
