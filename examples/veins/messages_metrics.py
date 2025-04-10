import ast

TIME_STEP = 5
EXPERIMENT = 'FMNIST_wscc'
METHOD = 'HVCFLThreshold'
ENVIRONMENT = 'final/8RSU/'
LOGS_PATH = 'python/experiments/' + EXPERIMENT + '/' + ENVIRONMENT + METHOD + '/1/logs/logs.txt'
THRESHOLD = 0.85
NUMBER_OF_VEHICLES = 100
NUMBER_OF_RSUS = 8

found_threshold = False
train_history = {'v' + str(i): {
    'valid_balanced_accuracy': [0.0]
} for i in range(NUMBER_OF_VEHICLES)}
chart_valid_accs = []
time = TIME_STEP
total_messages = 0
with open(LOGS_PATH, 'r') as logs_file:
    for line in logs_file.readlines():
        logs_data = ast.literal_eval(line)
        if logs_data['event'] == 'train':
            sim_time = logs_data['sim_time']
            node_id = logs_data['node_id']
            history = logs_data['history']
            if sim_time >= time:
                if len(train_history) > 0:
                    valid_accs = []
                    for h in train_history.values():
                        valid_accs.append(h['valid_balanced_accuracy'][-1])
                    chart_valid_accs.append(sum(valid_accs) / len(valid_accs))
                    if not found_threshold and chart_valid_accs[-1] >= THRESHOLD:
                        found_threshold = True
                        print('Threshold of ' + str(THRESHOLD) + ' found at time ' + str(time) + ' with ' + str(total_messages) + ' messages')
                time += TIME_STEP
            train_history[node_id] = history
        elif logs_data['event'] == 'aggregation':
            models = 1
            if 'number_of_clusters' in logs_data.keys():
                models += logs_data['number_of_clusters']
            total_messages += models * NUMBER_OF_RSUS
        elif logs_data['event'] == 'get_weights' and logs_data['node_id'] != 'server':
            total_messages += 1

valid_accs = []
for h in train_history.values():
    valid_accs.append(h['valid_balanced_accuracy'][-1])

print('Valid accuracy: ', sum(valid_accs) / len(valid_accs))
print('Total messages: ', total_messages)
