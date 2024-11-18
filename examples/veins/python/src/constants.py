from tensorflow import keras

SEED = 12
SPLIT = 0
EPOCHS = 1
BATCH_SIZE = 50
DATA_AUGMENTATION = True
LEARNING_RATE = 0.001
keras.utils.set_random_seed(SEED)

MNIST = 'MNIST'
FMNIST = 'FMNIST'
CIFAR10 = 'CIFAR10'
FEMNIST = 'FEMNIST'
GTSRB = 'GTSRB'
DATASET = FMNIST

GOSSIP_LEARNING = 'Gossip Learning'
OUR_METHOD = 'Our Method'
FED_AVG = 'FedAvg'
FED_PROX = 'FedProx'
WSCC = 'WSCC'
HYBRID_METHOD = 'Hybrid Method'
FED_PC = 'FedPC'
EXPERIMENT = HYBRID_METHOD

DATA_PATH = 'python/' + DATASET + '/data/'
WEIGHTS_FOLDER = 'weights/'
WEIGHTS_FILE_SUFFIX = '_weights.pickle'
LOGS_PATH = 'logs/logs.txt'
DATA_GENERATED_FOLDER = 'data/'
TMP_FOLDER = 'tmp/'
TMP_FILE_SUFFIX = '.pickle'

THRESHOLD = 0.1

CLEAR_TIME = 5

ENABLE_TMP_FOLDER = True
