import os
import ast
import pickle
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from sklearn.metrics import balanced_accuracy_score
from python.src import models

TIME_STEP = 5
EXPERIMENT = 'FMNIST_rot4'
METHOD = 'GossipLearning'
ENVIRONMENT = 'final/8RSU/'
REPLACE = 'REPLACE'
LOGS_PATH = 'python/experiments/' + EXPERIMENT + '/' + ENVIRONMENT + METHOD + '/' + REPLACE + '/logs/logs.txt'
DATA_PATH = 'python/' + EXPERIMENT.split('_')[0] + '/data/'
WEIGHTS_PATH = 'python/experiments/' + EXPERIMENT + '/' + ENVIRONMENT + METHOD + '/' + REPLACE + '/weights/'
THRESHOLD = 0.85
NUMBER_OF_VEHICLES = 100
NUMBER_OF_EXPERIMENTS = 1

final_train_accs = []
final_valid_accs = []
final_train_losses = []
final_valid_losses = []
final_train_acc = []
final_valid_acc = []
final_train_loss = []
final_valid_loss = []
for exp in range(NUMBER_OF_EXPERIMENTS):
    train_logs = []
    with open(LOGS_PATH.replace(REPLACE, str(exp + 1)), 'r') as logs_file:
        for line in logs_file.readlines():
            logs_data = ast.literal_eval(line)
            if logs_data['event'] == 'train':
                train_logs.append(logs_data)

    found_threshold = False
    train_history = {'v' + str(i): {
        'train_balanced_accuracy': [0.0],
        'valid_balanced_accuracy': [0.0]
    } for i in range(NUMBER_OF_VEHICLES)}
    chart_train_accs = []
    chart_valid_accs = []
    chart_train_losses = []
    chart_valid_losses = []
    chart_times = []
    time = TIME_STEP
    for tl in train_logs:
        sim_time = tl['sim_time']
        node_id = tl['node_id']
        history = tl['history']
        if sim_time >= time:
            if len(train_history) > 0:
                train_accs = []
                valid_accs = []
                train_losses = []
                valid_losses = []
                for h in train_history.values():
                    train_accs.append(h['train_balanced_accuracy'][-1])
                    valid_accs.append(h['valid_balanced_accuracy'][-1])
                    if 'loss' in h:
                        train_losses.append(h['loss'][-1])
                        valid_losses.append(h['val_loss'][-1])
                chart_train_accs.append(sum(train_accs) / len(train_accs))
                chart_valid_accs.append(sum(valid_accs) / len(valid_accs))
                if len(train_losses) > 0:
                    chart_train_losses.append(sum(train_losses) / len(train_losses))
                    chart_valid_losses.append(sum(valid_losses) / len(valid_losses))
                chart_times.append(time)
                if not found_threshold and chart_valid_accs[-1] >= THRESHOLD:
                    found_threshold = True
                    print('Threshold of ' + str(THRESHOLD) + ' found at time ' + str(time))
            time += TIME_STEP
        train_history[node_id] = history

    train_accs = []
    valid_accs = []
    train_losses = []
    valid_losses = []
    for h in train_history.values():
        train_accs.append(h['train_balanced_accuracy'][-1])
        valid_accs.append(h['valid_balanced_accuracy'][-1])
        if 'loss' in h:
            train_losses.append(h['loss'][-1])
            valid_losses.append(h['val_loss'][-1])

    final_train_accs.append(chart_train_accs)
    final_valid_accs.append(chart_valid_accs)
    final_train_losses.append(chart_train_losses)
    final_valid_losses.append(chart_valid_losses)

    final_train_acc.append(sum(train_accs) / len(train_accs))
    final_valid_acc.append(sum(valid_accs) / len(valid_accs))
    final_train_loss.append(sum(train_losses) / len(train_losses))
    final_valid_loss.append(sum(valid_losses) / len(valid_losses))

print('Train accuracy: ', final_train_acc)
print('Valid accuracy: ', final_valid_acc)
print('Train loss: ', final_train_loss)
print('Valid loss: ', final_valid_loss)

print('Train accuracy: ', sum(final_train_acc) / len(final_train_acc))
print('Valid accuracy: ', sum(final_valid_acc) / len(final_valid_acc))
print('Train loss: ', sum(final_train_loss) / len(final_train_loss))
print('Valid loss: ', sum(final_valid_loss) / len(final_valid_loss))

size = 1000000
for data in final_train_accs:
    if len(data) < size:
        size = len(data)
chart_times = chart_times[:size]
for i in range(len(final_train_accs)):
    final_train_accs[i] = final_train_accs[i][:size]
    final_valid_accs[i] = final_valid_accs[i][:size]
    final_train_losses[i] = final_train_losses[i][:size-1]
    final_valid_losses[i] = final_valid_losses[i][:size-1]

plt.figure()
plt.plot(chart_times, np.average(np.array(final_train_accs), axis=0), label='train')
plt.plot(chart_times, np.average(np.array(final_valid_accs), axis=0), label='validation')
plt.xlabel('Time')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('results/' + EXPERIMENT + '_' + METHOD + '_acc.png')
plt.close()

plt.figure()
plt.plot(chart_times[1:], np.average(np.array(final_train_losses), axis=0), label='train')
plt.plot(chart_times[1:], np.average(np.array(final_valid_losses), axis=0), label='validation')
plt.xlabel('Time')
plt.ylabel('Loss')
plt.legend()
plt.savefig('results/' + EXPERIMENT + '_' + METHOD + '_loss.png')
plt.close()

X_test_separated = {}
y_test_separated = {}
X_test_united = {}
y_test_united = {}
vehicle_clusters = {}
for file in os.listdir(DATA_PATH):
    vehicle = file.split('_')[0]
    cluster = file.split('_')[1]
    vehicle_clusters[vehicle] = cluster
    data = np.load(DATA_PATH + file)
    num_classes = data['num_classes']
    if cluster not in X_test_united.keys():
        X_test_united[cluster] = data['images_test']
        y_test_united[cluster] = keras.utils.to_categorical(data['labels_test'], num_classes)
    else:
        X_test_united[cluster] = np.concatenate((X_test_united[cluster], data['images_test']))
        y_test_united[cluster] = np.concatenate((y_test_united[cluster], keras.utils.to_categorical(data['labels_test'], num_classes)))
    X_test_separated[vehicle] = data['images_test']
    y_test_separated[vehicle] = keras.utils.to_categorical(data['labels_test'], num_classes)

results_separated = []
results_united = []
for exp in range(NUMBER_OF_EXPERIMENTS):
    print('Experiment ' + str(exp + 1))
    model = models.get_model()
    result_separated = []
    result_united = []
    for file in os.listdir(WEIGHTS_PATH.replace(REPLACE, str(exp + 1))):
        with open(WEIGHTS_PATH.replace(REPLACE, str(exp + 1)) + file, 'rb') as f:
            vehicle = file.split('_')[0]
            print('Vehicle ' + vehicle)
            if vehicle != 'server':
                cluster = vehicle_clusters[vehicle]
                weights = pickle.load(f)
                model.set_weights(weights)
                y_pred = np.argmax(model.predict(X_test_separated[vehicle]), axis=1)
                y_true = np.argmax(y_test_separated[vehicle], axis=1)
                balanced_accuracy = balanced_accuracy_score(y_true, y_pred)
                result_separated.append(balanced_accuracy)
                y_pred = np.argmax(model.predict(X_test_united[cluster]), axis=1)
                y_true = np.argmax(y_test_united[cluster], axis=1)
                balanced_accuracy = balanced_accuracy_score(y_true, y_pred)
                result_united.append(balanced_accuracy)
    results_separated.append(sum(result_separated) / len(result_separated))
    results_united.append(sum(result_united) / len(result_united))

print('Test accuracy separated: ', results_separated)
print('Test accuracy united: ', results_united)

print('Test accuracy separated: ', sum(results_separated) / len(results_separated))
print('Test accuracy united: ', sum(results_united) / len(results_united))
