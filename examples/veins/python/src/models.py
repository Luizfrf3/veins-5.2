import os
import gc
import pickle
import tensorflow as tf
from tensorflow.python import keras
from tensorflow.python.keras import layers
from tensorflow.python.ops import standard_ops
from tensorflow.python.keras import optimizer_v2
from sklearn.metrics import balanced_accuracy_score
from python.src import constants

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

class BalancedAccuracyCallback(keras.callbacks.Callback):

    def __init__(self, X_train, y_train, X_valid, y_valid):
        super(BalancedAccuracyCallback, self).__init__()
        self.X_train = X_train
        self.y_train = y_train
        self.X_valid = X_valid
        self.y_valid = y_valid

    def on_epoch_end(self, epoch, logs={}):
        y_pred = tf.argmax(self.model.predict(self.X_train), axis=1)
        y_true = tf.argmax(self.y_train, axis=1)
        train_balanced_accuracy = balanced_accuracy_score(y_true, y_pred)
        logs["train_balanced_accuracy"] = train_balanced_accuracy

        y_pred = tf.argmax(self.model.predict(self.X_valid), axis=1)
        y_true = tf.argmax(self.y_valid, axis=1)
        valid_balanced_accuracy = balanced_accuracy_score(y_true, y_pred)
        logs["valid_balanced_accuracy"] = valid_balanced_accuracy

def get_model():
    model = None

    # FLIS: MNIST, FMNIST, CIFAR10
    # MicronNet: FEMNIST, GTSRB
    if constants.DATASET in (constants.MNIST, constants.FMNIST):
        model = keras.Sequential(
            [
                keras.Input(shape=(28, 28, 1)),
                layers.Conv2D(6, kernel_size=(5, 5), padding="same", name="conv0", activation="relu"),
                layers.MaxPooling2D(pool_size=(2, 2)),
                layers.Conv2D(16, kernel_size=(5, 5), padding="same", name="conv1", activation="relu"),
                layers.MaxPooling2D(pool_size=(2, 2)),
                layers.Flatten(),
                layers.Dropout(0.25),
                layers.Dense(120, name="dense0", activation="relu"),
                layers.Dropout(0.5),
                layers.Dense(84, name="dense1", activation="relu"),
                layers.Dropout(0.5),
                layers.Dense(10, name="dense2", activation="softmax")
            ]
        )
    elif constants.DATASET == constants.CIFAR10:
        model = keras.Sequential(
            [
                keras.Input(shape=(32, 32, 3)),
                layers.Conv2D(6, kernel_size=(5, 5), padding="same", name="conv0", activation="relu"),
                layers.MaxPooling2D(pool_size=(2, 2)),
                layers.Conv2D(16, kernel_size=(5, 5), padding="same", name="conv1", activation="relu"),
                layers.MaxPooling2D(pool_size=(2, 2)),
                layers.Flatten(),
                layers.Dropout(0.25),
                layers.Dense(120, name="dense0", activation="relu"),
                layers.Dropout(0.5),
                layers.Dense(84, name="dense1", activation="relu"),
                layers.Dropout(0.5),
                layers.Dense(10, name="dense2", activation="softmax")
            ]
        )
    elif constants.DATASET == constants.FEMNIST:
        model = keras.Sequential(
            [
                keras.Input(shape=(28, 28, 1)),
                layers.Conv2D(29, kernel_size=(5, 5), padding="same", name="conv0", activation="relu"),
                layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)),
                layers.Conv2D(59, kernel_size=(3, 3), padding="same", name="conv1", activation="relu"),
                layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)),
                layers.Conv2D(74, kernel_size=(3, 3), padding="same", name="conv2", activation="relu"),
                layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)),
                layers.Flatten(),
                layers.Dropout(0.5),
                layers.Dense(300, name="dense0", activation="relu"),
                layers.Dropout(0.5),
                layers.Dense(62, name="dense1", activation="softmax")
            ]
        )
    elif constants.DATASET == constants.GTSRB:
        model = keras.Sequential(
            [
                keras.Input(shape=(48, 48, 3)),
                layers.Conv2D(29, kernel_size=(5, 5), padding="same", name="conv0", activation="relu"),
                layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)),
                layers.Conv2D(59, kernel_size=(3, 3), padding="same", name="conv1", activation="relu"),
                layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)),
                layers.Conv2D(74, kernel_size=(3, 3), padding="same", name="conv2", activation="relu"),
                layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)),
                layers.Flatten(),
                layers.Dropout(0.5),
                layers.Dense(300, name="dense0", activation="relu"),
                layers.Dropout(0.5),
                layers.Dense(43, name="dense1", activation="softmax")
            ]
        )

    if constants.EXPERIMENT == constants.FED_PROX or constants.EXPERIMENT == constants.FED_PC:
        model.compile(loss="categorical_crossentropy", optimizer=FedProxOptimizer(learning_rate=constants.LEARNING_RATE, mu=constants.MU), metrics=["accuracy"])
    else:
        model.compile(loss="categorical_crossentropy", optimizer=optimizer_v2.adam.Adam(learning_rate=constants.LEARNING_RATE), metrics=["accuracy"])
        #model.compile(loss="categorical_crossentropy", optimizer=optimizer_v2.gradient_descent.SGD(learning_rate=constants.LEARNING_RATE), metrics=["accuracy"])

    return model

def get_outputs(model):
    outputs = None

    if constants.DATASET in (constants.CIFAR10, constants.MNIST, constants.FMNIST):
        outputs = [
            model.get_layer("conv0").output,
            model.get_layer("conv1").output,
            model.get_layer("dense0").output,
            model.get_layer("dense1").output,
            model.get_layer("dense2").output
        ]
    elif constants.DATASET in (constants.FEMNIST, constants.GTSRB):
        outputs = [
            model.get_layer("conv0").output,
            model.get_layer("conv1").output,
            model.get_layer("conv2").output,
            model.get_layer("dense0").output,
            model.get_layer("dense1").output
        ]

    return outputs

def save_weights(node_id, weights):
    weights_path = constants.WEIGHTS_FOLDER + node_id + constants.WEIGHTS_FILE_SUFFIX
    with open(weights_path, 'wb') as weights_file:
        pickle.dump(weights, weights_file)

def encode_weights(weights, node_id):
    if constants.ENABLE_TMP_FOLDER:
        weights_path = constants.TMP_FOLDER + str(node_id) + constants.TMP_FILE_SUFFIX
        with open(weights_path, 'wb') as weights_file:
            pickle.dump(weights, weights_file)
        return ''
    else:
        weights_bytes = pickle.dumps(weights)
        raw_weights = ''
        for byte in weights_bytes:
            raw_weights += str(byte) + ','
        return raw_weights[:-1]

def decode_weights(raw_weights, sender_id):
    if constants.ENABLE_TMP_FOLDER:
        weights_path = constants.TMP_FOLDER + str(sender_id) + constants.TMP_FILE_SUFFIX
        with open(weights_path, 'rb') as weights_file:
            return pickle.load(weights_file)
    else:
        byte_list = []
        for byte_str in raw_weights.split(','):
            byte_list.append(int(byte_str))
        weights_bytes = bytes(byte_list)
        return pickle.loads(weights_bytes)

def clear_session():
    keras.backend.clear_session()
    gc.collect()

@tf.keras.utils.register_keras_serializable()
class FedProxOptimizer(optimizer_v2.optimizer_v2.OptimizerV2):

    def __init__(self, learning_rate=0.01, mu=0.001, name='FedProxOptimizer', **kwargs):
        super().__init__(name=name, **kwargs)

        self._set_hyper('learning_rate', learning_rate)
        self._set_hyper('mu', mu)

        self._lr_t = None
        self._mu_t = None

    def _prepare(self, var_list):
        self._lr_t = tf.convert_to_tensor(self._get_hyper('learning_rate'), name='lr')
        self._mu_t = tf.convert_to_tensor(self._get_hyper('mu'), name='mu')

    def _create_slots(self, var_list):
        for v in var_list:
            self.add_slot(v, 'vstar')

    def _resource_apply_dense(self, grad, var):
        lr_t = tf.cast(self._lr_t, var.dtype.base_dtype)
        mu_t = tf.cast(self._mu_t, var.dtype.base_dtype)
        vstar = self.get_slot(var, 'vstar')

        var_update = var.assign_sub(lr_t * (grad + mu_t * (var - vstar)))

        return tf.group(*[var_update, ])

    def _apply_sparse_shared(self, grad, var, indices, scatter_add):
        lr_t = tf.cast(self._lr_t, var.dtype.base_dtype)
        mu_t = tf.cast(self._mu_t, var.dtype.base_dtype)
        vstar = self.get_slot(var, 'vstar')
        v_diff = vstar.assign(mu_t * (var - vstar), use_locking=self._use_locking)

        with tf.control_dependencies([v_diff]):
            scaled_grad = scatter_add(vstar, indices, grad)
        var_update = var.assign_sub(lr_t * scaled_grad)

        return tf.group(*[var_update, ])

    def _resource_apply_sparse(self, grad, var):
        return self._apply_sparse_shared(
            grad.values, var, grad.indices,
            lambda x, i, v: standard_ops.scatter_add(x, i, v))

    def get_config(self):
        base_config = super(FedProxOptimizer, self).get_config()
        return {
            **base_config,
            'lr': self._serialize_hyperparameter('learning_rate'),
            'mu': self._serialize_hyperparameter('mu')
        }
