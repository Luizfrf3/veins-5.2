import ast
import matplotlib.pyplot as plt

TIME_STEP = 5
EXPERIMENT = 'CIFAR10_rot2'
METHOD = 'GossipLearning'
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
                train_accs.append(h['accuracy'][-1])
                valid_accs.append(h['val_accuracy'][-1])
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
    train_accs.append(h['accuracy'][-1])
    valid_accs.append(h['val_accuracy'][-1])
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
