from python.src import constants

def register_log(logs_data):
    with open(constants.LOGS_PATH, 'a') as f:
        f.write('{}\n'.format(logs_data))
