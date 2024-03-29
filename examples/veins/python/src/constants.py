from tensorflow import keras

SEED = 12
SPLIT = 0
EPOCHS = 5
BATCH_SIZE = 50
DATA_AUGMENTATION = True
keras.utils.set_random_seed(SEED)

MNIST = 'MNIST'
CIFAR10 = 'CIFAR10'
FEMNIST = 'FEMNIST'
GTSRB = 'GTSRB'
DATASET = CIFAR10

GOSSIP_LEARNING = 'Gossip Learning'
OUR_METHOD = 'Our Method'
FED_AVG = 'FedAvg'
FED_PROX = 'FedProx'
WSCC = 'WSCC'
EXPERIMENT = OUR_METHOD

DATA_PATH = 'python/' + DATASET + '/data/'
WEIGHTS_FOLDER = 'weights/'
WEIGHTS_FILE_SUFFIX = '_weights.pickle'
LOGS_PATH = 'logs/logs.txt'

THRESHOLD = 0.02
