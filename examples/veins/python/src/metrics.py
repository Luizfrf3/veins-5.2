import numpy as np
import tensorflow as tf
from sklearn.cross_decomposition import CCA
from sklearn.metrics import balanced_accuracy_score

def flatten(w):
    r = np.array([], dtype=np.float32)
    for i in range(len(w)):
        r = np.concatenate((r, w[i].flatten()), axis=0)
    return r

def cossim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def cka(features_x, features_y):
    features_x = features_x - np.mean(features_x, 0, keepdims=True)
    features_y = features_y - np.mean(features_y, 0, keepdims=True)
    dot_product_similarity = np.linalg.norm(features_x.T.dot(features_y)) ** 2
    normalization_x = np.linalg.norm(features_x.T.dot(features_x))
    normalization_y = np.linalg.norm(features_y.T.dot(features_y))
    return dot_product_similarity / (normalization_x * normalization_y)

#def cca(features_x, features_y):
#    qx, _ = np.linalg.qr(features_x)
#    qy, _ = np.linalg.qr(features_y)
#    return np.linalg.norm(qx.T.dot(qy)) ** 2 / min(features_x.shape[1], features_y.shape[1])

def cca(features_x, features_y, n_components):
    cca_obj = CCA(n_components=n_components, max_iter=2000)
    cca_obj.fit(features_x, features_y)
    x_c, y_c = cca_obj.transform(features_x, features_y)
    return np.mean([np.corrcoef(x_c[:, i], y_c[:, i])[0, 1] for i in range(n_components)])

def balanced_accuracy(model, rmodel, X, y):
    y_true = tf.argmax(y, axis=1)
    maccuracy = balanced_accuracy_score(y_true, tf.argmax(model.predict(X), axis=1))
    raccuracy = balanced_accuracy_score(y_true, tf.argmax(rmodel.predict(X), axis=1))
    return maccuracy, raccuracy
