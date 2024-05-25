import os
import ast
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import balanced_accuracy_score
from python.src import models

TIME_STEP = 5
EXPERIMENT = 'FMNIST_dir'
METHOD = 'FedAvg'
LOGS_PATH = 'python/experiments/' + EXPERIMENT + '/' + METHOD + '/logs/logs.txt'

train_logs = []
with open(LOGS_PATH, 'r') as logs_file:
    for line in logs_file.readlines():
        logs_data = ast.literal_eval(line)
        if logs_data['event'] == 'train':
            train_logs.append(logs_data)

train_history = {}
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
                train_losses.append(h['loss'][-1])
                valid_losses.append(h['val_loss'][-1])
            chart_train_accs.append(sum(train_accs) / len(train_accs))
            chart_valid_accs.append(sum(valid_accs) / len(valid_accs))
            chart_train_losses.append(sum(train_losses) / len(train_losses))
            chart_valid_losses.append(sum(valid_losses) / len(valid_losses))
            chart_times.append(time)
        time += TIME_STEP
    train_history[node_id] = history

train_accs = []
valid_accs = []
train_losses = []
valid_losses = []
for h in train_history.values():
    train_accs.append(h['train_balanced_accuracy'][-1])
    valid_accs.append(h['valid_balanced_accuracy'][-1])
    train_losses.append(h['loss'][-1])
    valid_losses.append(h['val_loss'][-1])

print('Train accuracy: ', sum(train_accs) / len(train_accs))
print('Valid accuracy: ', sum(valid_accs) / len(valid_accs))
print('Train loss: ', sum(train_losses) / len(train_losses))
print('Valid loss: ', sum(valid_losses) / len(valid_losses))

plt.figure()
plt.plot(chart_times, chart_train_accs, label='train')
plt.plot(chart_times, chart_valid_accs, label='validation')
plt.xlabel('Time')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('results/' + EXPERIMENT + '_' + METHOD + '_acc.png')
plt.close()

plt.figure()
plt.plot(chart_times, chart_train_losses, label='train')
plt.plot(chart_times, chart_valid_losses, label='validation')
plt.xlabel('Time')
plt.ylabel('Loss')
plt.legend()
plt.savefig('results/' + EXPERIMENT + '_' + METHOD + '_loss.png')
plt.close()

'''
DATA_PATH = 'python/experiments/' + EXPERIMENT + '/' + METHOD + '/data/'
X_valid = {}
y_valid = {}
vehicle_clusters = {}
for file in os.listdir(DATA_PATH):
    vehicle = file.split('_')[0]
    cluster = file.split('_')[1]
    vehicle_clusters[vehicle] = cluster
    data = np.load(DATA_PATH + file)
    if cluster not in X_valid.keys():
        X_valid[cluster] = data['X_valid']
        y_valid[cluster] = data['y_valid']
    else:
        X_valid[cluster] = np.concatenate((X_valid[cluster], data['X_valid']))
        y_valid[cluster] = np.concatenate((y_valid[cluster], data['y_valid']))

WEIGHTS_PATH = 'python/experiments/' + EXPERIMENT + '/' + METHOD + '/weights/'
model = models.get_model()
result = []
for file in os.listdir(WEIGHTS_PATH):
    with open(WEIGHTS_PATH + file, 'rb') as f:
        vehicle = file.split('_')[0]
        if vehicle != 'server':
            cluster = vehicle_clusters[vehicle]
            weights = pickle.load(f)
            model.set_weights(weights)
            y_pred = np.argmax(model.predict(X_valid[cluster]), axis=1)
            y_true = np.argmax(y_valid[cluster], axis=1)
            balanced_accuracy = balanced_accuracy_score(y_true, y_pred)
            result.append(balanced_accuracy)
print('Valid accuracy overall: ', sum(result) / len(result))
'''
