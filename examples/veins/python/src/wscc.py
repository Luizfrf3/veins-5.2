import logging
import random
import numpy as np
from sklearn.cluster import AffinityPropagation
from python.src import constants, models, logs, metrics
random.seed(constants.SEED)

received_weights = {}
dataset_sizes = {}

participating_nodes = {}
clusters_nodes = {}
clusters_weights = {}

def _cluster_aggregation(node_id, model):
    nodes_data = list(received_weights[node_id].items())
    benchmark = random.randrange(len(nodes_data))
    bw = metrics.flatten(nodes_data[benchmark][1])

    cossims = []
    for sender, rweight in nodes_data:
        cossim = metrics.cossim(bw, metrics.flatten(rweight))
        cossims.append([cossim])

    clustering = AffinityPropagation().fit(cossims)

    clusters_data = {}
    for i in range(len(clustering.labels_)):
        cluster = clustering.labels_[i]
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

    return len(clustering.labels_)

def _global_aggregation(node_id, model):
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

    return weights

def store_weights(raw_weights, dataset_size, node_id, sender_id):
    weights = models.decode_weights(raw_weights)
    if node_id not in received_weights.keys():
        received_weights[node_id] = {}
        dataset_sizes[node_id] = {}
    received_weights[node_id][sender_id] = weights
    dataset_sizes[node_id][sender_id] = dataset_size

def aggregation(aggregation_round, node_id, number_of_received_models, sim_time, node_models):
    model = node_models[node_id]

    number_of_clusters = 0
    participating_nodes[node_id] = []
    clusters_nodes[node_id] = {}
    clusters_weights[node_id] = {}

    if node_id in received_weights.keys() and len(received_weights[node_id]) > 0:
        participating_nodes[node_id] = list(received_weights[node_id].keys())
        number_of_clusters = _cluster_aggregation(node_id, model)
        weights = _global_aggregation(node_id, model)
        model.set_weights(weights)

    logs_data = {'event': 'aggregation', 'node_id': node_id, 'sim_time': sim_time, 'aggregation_round': aggregation_round, 'number_of_received_models': number_of_received_models, 'number_of_clusters': number_of_clusters, 'participating_nodes': participating_nodes, 'clusters_nodes': clusters_nodes}
    logs.register_log(logs_data)

    models.save_weights(node_id, model.get_weights())
    node_models[node_id] = model

    received_weights[node_id] = {}
    dataset_sizes[node_id] = {}

    return number_of_clusters

def get_participating_nodes(node_id, sim_time):
    return ','.join(participating_nodes[node_id])

def get_cluster_weights(node_id, cluster, sim_time):
    return models.encode_weights(clusters_weights[node_id][cluster])

def get_cluster_nodes(node_id, cluster, sim_time):
    return ','.join(clusters_nodes[node_id][cluster])

def train(node_id, training_round, sim_time, vehicle_data, node_models):
    X_train, y_train = vehicle_data[node_id]['train']
    X_valid, y_valid = vehicle_data[node_id]['valid']

    accepted_model = False
    model = node_models[node_id]
    rmodel = models.get_model()
    rmodel.set_weights(rweight)
    _, maccuracy = model.evaluate(X_valid, y_valid, verbose=0)
    _, raccuracy = rmodel.evaluate(X_valid, y_valid, verbose=0)
    if raccuracy >= maccuracy or abs(maccuracy - raccuracy) <= constants.THRESHOLD:
        model = rmodel
        accepted_model = True

    history = model.fit(X_train, y_train, epochs=constants.EPOCHS, validation_data=(X_valid, y_valid), verbose=0)

    logging.warning('Node {}, Training Round {}, Accepted Model {}, History {}'.format(node_id, training_round, accepted_model, history.history))
    logs_data = {'event': 'train', 'node_id': node_id, 'sim_time': sim_time, 'training_round': training_round, 'accepted_model': accepted_model, 'history': history.history}
    logs.register_log(logs_data)

    models.save_weights(node_id, model.get_weights())
    node_models[node_id] = model