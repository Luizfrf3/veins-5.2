import torch
import numpy as np
import tensorflow as tf
from sklearn.cross_decomposition import CCA
from sklearn.metrics import balanced_accuracy_score

device = torch.device('cuda')

def flatten(w):
    r = np.array([], dtype=np.float32)
    for i in range(len(w)):
        r = np.concatenate((r, w[i].flatten()), axis=0)
    return r

def cossim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def centering(K):
    K = torch.tensor([K], device=device) if len(K.shape) == 0 else K
    n = K.shape[0]
    unit = torch.ones([n, n], device=device)
    I = torch.eye(n, device=device)
    H = I - unit / n
    return torch.matmul(torch.matmul(H, K), H)

def hsic(X, Y):
    L_X = torch.matmul(X, X.T)
    L_Y = torch.matmul(Y, Y.T)
    return torch.sum(centering(L_X) * centering(L_Y))

def cka(X, Y):
    X = torch.from_numpy(X).cuda()
    Y = torch.from_numpy(Y).cuda()
    return (hsic(X, Y) / (torch.sqrt(hsic(X, X)) * torch.sqrt(hsic(Y, Y)))).cpu()

#def cca(X, Y):
#    qx, _ = np.linalg.qr(X)
#    qy, _ = np.linalg.qr(Y)
#    return np.linalg.norm(qx.T.dot(qy)) ** 2 / min(X.shape[1], Y.shape[1])

def cca(X, Y, n_components):
    cca_obj = CCA(n_components=n_components, max_iter=2000)
    cca_obj.fit(X, Y)
    x_c, y_c = cca_obj.transform(X, Y)
    return np.mean([np.corrcoef(x_c[:, i], y_c[:, i])[0, 1] for i in range(n_components)])

def balanced_accuracy(model, rmodel, X, y):
    y_true = tf.argmax(y, axis=1)
    maccuracy = balanced_accuracy_score(y_true, tf.argmax(model.predict(X), axis=1))
    raccuracy = balanced_accuracy_score(y_true, tf.argmax(rmodel.predict(X), axis=1))
    return maccuracy, raccuracy
