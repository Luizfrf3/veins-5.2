import logging
import copy
import math
import random
import numpy as np
from tensorflow.python import keras
from keras.preprocessing.image import ImageDataGenerator
from sklearn.cluster import AffinityPropagation
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.signal import argrelextrema
from python.src import constants, models, logs, metrics
random.seed(constants.SEED)

received_weights = {}
received_weights_while_training = {}
dataset_sizes = {}
dataset_sizes_while_training = {}
received_model_from_server = {}

participating_nodes = {}
clusters_nodes = {}
clusters_weights = {}

rmodel = models.get_model()

clean_time = [constants.CLEAR_TIME]

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

def _weighted_aggregation(node_id, model, mw, X_valid, y_valid):
    #loss, accuracy = model.evaluate(X_valid, y_valid, verbose=0)
    #inter_model = keras.Model(inputs=model.input, outputs=models.get_outputs(model))
    #activations = [_preprocess_activations(act) for act in inter_model(X_valid)]
    #mfeatures = [1.0 for _ in range(len(activations))]
    mfeatures = [1.0]
    #mfeatures = [1.0, 1.0, 1.0, loss, accuracy]
    #mfeatures = [metrics.cca(act, act, act.shape[1]) for act in activations]

    result = []
    for sender, rweight in received_weights[node_id].items():
        #loss, accuracy = rmodel.evaluate(X_valid, y_valid, verbose=0)
        cossim = metrics.cossim(mw, metrics.flatten(rweight))
        #rmodel.set_weights(rweight)
        #rinter_model = keras.Model(inputs=rmodel.input, outputs=models.get_outputs(rmodel))
        #ractivations = [_preprocess_activations(ract) for ract in rinter_model(X_valid)]
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

def _local_clustering(node_id, model, mw, X_valid, y_valid):
    #loss, accuracy = model.evaluate(X_valid, y_valid, verbose=0)
    #inter_model = keras.Model(inputs=model.input, outputs=models.get_outputs(model))
    #activations = [_preprocess_activations(act) for act in inter_model(X_valid)]
    #mfeatures = [1.0 for _ in range(len(activations))]
    mfeatures = [1.0]
    #mfeatures = [1.0, 1.0, 1.0, loss, accuracy]
    #mfeatures = [metrics.cca(act, act, act.shape[1]) for act in activations]

    rfeatures = []
    rweights = []
    senders = []
    for sender, rweight in received_weights[node_id].items():
        #rmodel.set_weights(rweight)

        #loss, accuracy = rmodel.evaluate(X_valid, y_valid, verbose=0)
        cossim = metrics.cossim(mw, metrics.flatten(rweight))
        #rinter_model = keras.Model(inputs=rmodel.input, outputs=models.get_outputs(rmodel))
        #ractivations = [_preprocess_activations(ract) for ract in rinter_model(X_valid)]
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

def _cluster_aggregation(node_id, model):
    nodes_data = list(received_weights[node_id].items())

    benchmark = random.randrange(len(nodes_data))
    bw = metrics.flatten(nodes_data[benchmark][1])
    sender_benchmark = nodes_data[benchmark][0]
    cossims = []
    for sender, rweight in nodes_data:
        #if sender == sender_benchmark:
        #    continue
        cossim = metrics.cossim(bw, metrics.flatten(rweight))
        cossims.append([cossim])
    clustering = AffinityPropagation(damping=0.7, max_iter=2000).fit(cossims)
    labels = list(clustering.labels_)
    #benchmark_cluster = clustering.predict([[1.0]])[0]
    #labels = np.insert(labels, benchmark, benchmark_cluster)

    #distances = []
    #for sender1, rweight1 in nodes_data:
    #    rw1 = metrics.flatten(rweight1)
    #    distancerow = []
    #    for sender2, rweight2 in nodes_data:
    #        cossim = metrics.cossim(rw1, metrics.flatten(rweight2))
    #        distancerow.append((1 - cossim) / 2)
    #    distances.append(distancerow)
    #clustering = AffinityPropagation(damping=0.7, max_iter=2000, affinity='precomputed').fit(distances)
    #labels = clustering.labels_

    #labels = []
    #for sender, rweight in nodes_data:
    #    sid = int(sender[1:])
    #    if sid < 30:
    #        labels.append(0)
    #    elif sid < 60:
    #        labels.append(1)
    #    elif sid < 90:
    #        labels.append(2)
    #    else:
    #        labels.append(3)

    clusters_data = {}
    for i in range(len(labels)):
        cluster = labels[i]
        if cluster not in clusters_nodes[node_id]:
            clusters_nodes[node_id][cluster] = []
        clusters_nodes[node_id][cluster].append(nodes_data[i][0])
        if cluster not in clusters_data:
            clusters_data[cluster] = []
        clusters_data[cluster].append(nodes_data[i])

    for cluster, cdata in clusters_data.items():
        cweights = []
        model_weights = model.get_weights()
        for i in range(len(model_weights)):
            cweights.append(np.zeros(model_weights[i].shape))

        for i in range(len(cweights)):
            size = 0
            for sender, rweights in cdata:
                cweights[i] += rweights[i] * dataset_sizes[node_id][sender]
                size += dataset_sizes[node_id][sender]
            cweights[i] = cweights[i] / size
        clusters_weights[node_id][cluster] = cweights

    return max(labels) + 1, sender_benchmark

def _global_aggregation(node_id, model):
    weights = []

    clusters_nodes[node_id][-1] = []

    model_weights = model.get_weights()
    for i in range(len(model_weights)):
        weights.append(np.zeros(model_weights[i].shape))

    for i in range(len(weights)):
        size = 0
        for sender, rweights in received_weights[node_id].items():
            weights[i] += rweights[i] * dataset_sizes[node_id][sender]
            size += dataset_sizes[node_id][sender]
        weights[i] = weights[i] / size
    clusters_weights[node_id][-1] = weights

    return weights

def store_weights(raw_weights, dataset_size, node_id, sender_id):
    weights = models.decode_weights(raw_weights, sender_id)
    if node_id not in received_weights.keys():
        received_weights[node_id] = {}
        dataset_sizes[node_id] = {}
        received_model_from_server[node_id] = False
    received_weights[node_id][sender_id] = weights
    dataset_sizes[node_id][sender_id] = dataset_size

def store_weights_while_training(raw_weights, dataset_size, node_id, sender_id):
    weights = models.decode_weights(raw_weights, sender_id)
    if node_id not in received_weights_while_training.keys():
        received_weights_while_training[node_id] = {}
        dataset_sizes_while_training[node_id] = {}
        received_model_from_server[node_id] = False
    received_weights_while_training[node_id][sender_id] = weights
    dataset_sizes_while_training[node_id][sender_id] = dataset_size

def aggregation(aggregation_round, node_id, sim_time, node_models):
    model = node_models[node_id]

    number_of_clusters = 0
    sender_benchmark = ''
    participating_nodes[node_id] = []
    clusters_nodes[node_id] = {}
    clusters_weights[node_id] = {}

    if node_id not in received_weights.keys():
        received_weights[node_id] = {}
        dataset_sizes[node_id] = {}
        received_model_from_server[node_id] = False

    if len(received_weights[node_id]) > 0:
        participating_nodes[node_id] = list(received_weights[node_id].keys())
        number_of_clusters, sender_benchmark = _cluster_aggregation(node_id, model)
        weights = _global_aggregation(node_id, model)
        model.set_weights(weights)

    logs_data = {'event': 'aggregation', 'node_id': node_id, 'sim_time': sim_time, 'aggregation_round': aggregation_round, 'number_of_received_models': len(received_weights[node_id]), 'number_of_clusters': number_of_clusters, 'sender_benchmark': sender_benchmark, 'participating_nodes': participating_nodes, 'clusters_nodes': clusters_nodes}
    logs.register_log(logs_data)

    models.save_weights(node_id, model.get_weights())
    node_models[node_id] = model

    received_weights[node_id] = {}
    dataset_sizes[node_id] = {}
    received_model_from_server[node_id] = False

    return number_of_clusters

def get_participating_nodes(node_id, sim_time):
    return ','.join(participating_nodes[node_id])

def get_cluster_weights(node_id, cluster, sim_time):
    return models.encode_weights(clusters_weights[node_id][cluster], cluster)

def get_cluster_nodes(node_id, cluster, sim_time):
    return ','.join(clusters_nodes[node_id][cluster])

def receive_global_model(raw_weights, node_id, sender_id, sim_time, node_models, vehicle_data):
    X_valid, y_valid = vehicle_data[node_id]['valid']

    if node_id not in received_weights.keys():
        received_weights[node_id] = {}
        dataset_sizes[node_id] = {}
        received_model_from_server[node_id] = False
    if node_id not in received_weights_while_training.keys():
        received_weights_while_training[node_id] = {}
        dataset_sizes_while_training[node_id] = {}
        received_model_from_server[node_id] = False

    accepted_model = False
    model = node_models[node_id]
    rweights = models.decode_weights(raw_weights, sender_id)
    #rmodel.set_weights(rweights)
    #_, maccuracy = model.evaluate(X_valid, y_valid, verbose=0)
    #_, raccuracy = rmodel.evaluate(X_valid, y_valid, verbose=0)
    #if raccuracy >= maccuracy or abs(maccuracy - raccuracy) <= constants.THRESHOLD:
    #    model.set_weights(rweights)
    #    received_weights[node_id] = {}
    #    dataset_sizes[node_id] = {}
    #    received_weights_while_training[node_id] = {}
    #    dataset_sizes_while_training[node_id] = {}
    #    received_model_from_server[node_id] = True
    #    accepted_model = True
    model.set_weights(rweights)
    received_weights[node_id] = {}
    dataset_sizes[node_id] = {}
    received_weights_while_training[node_id] = {}
    dataset_sizes_while_training[node_id] = {}
    received_model_from_server[node_id] = True
    maccuracy = 0
    raccuracy = 0
    node_models[node_id] = model

    logs_data = {'event': 'receive_global_model', 'node_id': node_id, 'sim_time': sim_time, 'sender_id': sender_id, 'accepted_model': accepted_model, 'maccuracy': maccuracy, 'raccuracy': raccuracy}
    logs.register_log(logs_data)

def train(node_id, training_round, sim_time, vehicle_data, node_models):
    X_train, y_train = vehicle_data[node_id]['train']
    X_valid, y_valid = vehicle_data[node_id]['valid']

    accepted_model = False
    model = node_models[node_id]
    mweights = model.get_weights()

    if node_id not in received_weights.keys():
        received_weights[node_id] = {}
        dataset_sizes[node_id] = {}
        received_model_from_server[node_id] = False
    if node_id not in received_weights_while_training.keys():
        received_weights_while_training[node_id] = {}
        dataset_sizes_while_training[node_id] = {}
        received_model_from_server[node_id] = False

    participating_nodes = []
    cluster_nodes = []

    if not received_model_from_server[node_id] and len(received_weights[node_id]) > 0:
        participating_nodes = [node for node in received_weights[node_id].keys()]

        clustered_weights = _local_clustering(node_id, model, metrics.flatten(mweights), X_valid, y_valid)
        cluster_nodes = [cw['id'] for cw in clustered_weights]
        for i in range(len(mweights)):
            sizes = len(X_train)
            mweights[i] = mweights[i] * sizes
            for cw in clustered_weights:
                mweights[i] = mweights[i] + (cw['w'][i] * dataset_sizes[node_id][cw['id']])
                sizes += dataset_sizes[node_id][cw['id']]
            mweights[i] = mweights[i] / sizes

        #clustered_weights, mf = _weighted_aggregation(node_id, model, metrics.flatten(mweights), X_valid, y_valid)
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

        #rmodel.set_weights(mweights)
        #_, maccuracy = model.evaluate(X_valid, y_valid, verbose=0)
        #_, raccuracy = rmodel.evaluate(X_valid, y_valid, verbose=0)
        #if raccuracy >= maccuracy or abs(maccuracy - raccuracy) <= constants.THRESHOLD:
        #    model.set_weights(mweights)
        #    accepted_model = True
        model.set_weights(mweights)

    if constants.DATA_AUGMENTATION:
        datagen = ImageDataGenerator(zoom_range=0.2, horizontal_flip=True)
        #datagen = ImageDataGenerator(zoom_range=0.2, rotation_range=45, width_shift_range=0.2, height_shift_range=0.2)
        datagen.fit(X_train)
        history = model.fit(datagen.flow(X_train, y_train, batch_size=constants.BATCH_SIZE), steps_per_epoch = constants.EPOCHS * X_train.shape[0] / 50, validation_data=(X_valid, y_valid), callbacks=[models.BalancedAccuracyCallback(X_train, y_train, X_valid, y_valid)], verbose=0)
    else:
        history = model.fit(X_train, y_train, epochs=constants.EPOCHS, batch_size=constants.BATCH_SIZE, validation_data=(X_valid, y_valid), callbacks=[models.BalancedAccuracyCallback(X_train, y_train, X_valid, y_valid)], verbose=0)

    logging.warning('Node {}, Training Round {}, History {}'.format(node_id, training_round, history.history))
    logs_data = {'event': 'train', 'node_id': node_id, 'sim_time': sim_time, 'accepted_model': accepted_model, 'training_round': training_round, 'number_of_received_models': len(received_weights[node_id]), 'history': history.history, 'participating_nodes': participating_nodes, 'cluster_nodes': cluster_nodes}
    logs.register_log(logs_data)

    models.save_weights(node_id, model.get_weights())
    node_models[node_id] = model

    received_weights[node_id] = copy.deepcopy(received_weights_while_training[node_id])
    dataset_sizes[node_id] = copy.deepcopy(dataset_sizes_while_training[node_id])
    received_weights_while_training[node_id] = {}
    dataset_sizes_while_training[node_id] = {}
    received_model_from_server[node_id] = False

    if sim_time >= clean_time[0]:
        logging.warning('Clearing Keras session')
        models.clear_session()
        clean_time[0] = clean_time[0] + constants.CLEAR_TIME
