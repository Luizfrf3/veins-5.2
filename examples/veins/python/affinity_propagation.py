import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, datasets
from sklearn.cluster import AffinityPropagation
from sklearn.cross_decomposition import CCA
tf.keras.utils.set_random_seed(42)

num_classes = 10
input_shape = (28, 28, 1)

def cnn():
    return keras.Sequential(
        [
            keras.Input(shape=input_shape),
            layers.Conv2D(32, kernel_size=(3, 3), padding="same", activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(64, kernel_size=(3, 3), padding="same", activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Dropout(0.25),
            layers.Flatten(),
            layers.Dense(128, name="first_dense", activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(num_classes, name="last_dense", activation="softmax")
        ]
    )

def flatten(w):
    r = np.array([], dtype=np.float32)
    for i in range(len(w)):
        r = np.concatenate((r, w[i].flatten()), axis=0)
    return r

def mse(a, b):
    return np.square(np.subtract(a, b)).mean()

def cossim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

model0 = cnn()
model0.summary()

w0 = flatten(model0.get_weights())
model1 = cnn()
w1 = flatten(model1.get_weights())
model2 = cnn()
w2 = flatten(model2.get_weights())
model3 = cnn()
w3 = flatten(model3.get_weights())
model4 = cnn()
w4 = flatten(model4.get_weights())

mse0 = mse(w0, w0)
mse1 = mse(w0, w1)
mse2 = mse(w0, w2)
mse3 = mse(w0, w3)
mse4 = mse(w0, w4)
print(mse0)
print(mse1)
print(mse2)
print(mse3)
print(mse4)

cossim0 = cossim(w0, w0)
cossim1 = cossim(w0, w1)
cossim2 = cossim(w0, w2)
cossim3 = cossim(w0, w3)
cossim4 = cossim(w0, w4)
print(cossim0)
print(cossim1)
print(cossim2)
print(cossim3)
print(cossim4)

(x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
x_train = x_train.astype("float32")[:20] / 255

inter_model0 = keras.Model(inputs=model0.input, outputs=[model0.get_layer("first_dense").output, model0.get_layer("last_dense").output])
inter_model1 = keras.Model(inputs=model1.input, outputs=[model1.get_layer("first_dense").output, model1.get_layer("last_dense").output])
inter_model2 = keras.Model(inputs=model2.input, outputs=[model2.get_layer("first_dense").output, model2.get_layer("last_dense").output])
inter_model3 = keras.Model(inputs=model3.input, outputs=[model3.get_layer("first_dense").output, model3.get_layer("last_dense").output])
inter_model4 = keras.Model(inputs=model4.input, outputs=[model4.get_layer("first_dense").output, model4.get_layer("last_dense").output])

def cka(features_x, features_y):
    features_x = features_x - np.mean(features_x, 0, keepdims=True)
    features_y = features_y - np.mean(features_y, 0, keepdims=True)
    dot_product_similarity = np.linalg.norm(features_x.T.dot(features_y)) ** 2
    normalization_x = np.linalg.norm(features_x.T.dot(features_x))
    normalization_y = np.linalg.norm(features_y.T.dot(features_y))
    return dot_product_similarity / (normalization_x * normalization_y)

def centering(K):
    n = K.shape[0]
    unit = np.ones([n, n])
    I = np.eye(n)
    H = I - unit / n
    return np.dot(np.dot(H, K), H)

def linear_HSIC(X, Y):
    L_X = np.dot(X, X.T)
    L_Y = np.dot(Y, Y.T)
    return np.sum(centering(L_X) * centering(L_Y))

def linear_CKA(X, Y):
    hsic = linear_HSIC(X, Y)
    var1 = np.sqrt(linear_HSIC(X, X))
    var2 = np.sqrt(linear_HSIC(Y, Y))
    return hsic / (var1 * var2)

def cca(features_x, features_y):
    qx, _ = np.linalg.qr(features_x)
    qy, _ = np.linalg.qr(features_y)
    return np.linalg.norm(qx.T.dot(qy)) ** 2 / min(features_x.shape[1], features_y.shape[1])

x0 = np.array(inter_model0(x_train)[1])
x1 = np.array(inter_model1(x_train)[1])
x2 = np.array(inter_model2(x_train)[1])
x3 = np.array(inter_model3(x_train)[1])
x4 = np.array(inter_model4(x_train)[1])

cka0 = cka(x0, x0)
cka1 = cka(x0, x1)
cka2 = cka(x0, x2)
cka3 = cka(x0, x3)
cka4 = cka(x0, x4)
print(cka0)
print(cka1)
print(cka2)
print(cka3)
print(cka4)

print(linear_CKA(x0, x0))
print(linear_CKA(x0, x1))
print(linear_CKA(x0, x2))
print(linear_CKA(x0, x3))
print(linear_CKA(x0, x4))

cca0 = cca(x0, x0)
cca1 = cca(x0, x1)
cca2 = cca(x0, x2)
cca3 = cca(x0, x3)
cca4 = cca(x0, x4)
print(cca0)
print(cca1)
print(cca2)
print(cca3)
print(cca4)

ccas = CCA(n_components=10, max_iter=2000)
print(ccas.fit(x0, x0).score(x0, x0))
print(ccas.fit(x0, x1).score(x0, x1))
print(ccas.fit(x0, x2).score(x0, x2))
print(ccas.fit(x0, x3).score(x0, x3))
print(ccas.fit(x0, x4).score(x0, x4))

from cca_zoo.linear import CCA
x0 -= x0.mean(axis=0)
x1 -= x1.mean(axis=0)
x2 -= x2.mean(axis=0)
x3 -= x3.mean(axis=0)
x4 -= x4.mean(axis=0)
ccas = CCA(latent_dimensions=10)
print(ccas.fit((x0, x0)).score((x0, x0)) / 10)
print(ccas.fit((x0, x1)).score((x0, x1)) / 10)
print(ccas.fit((x0, x2)).score((x0, x2)) / 10)
print(ccas.fit((x0, x3)).score((x0, x3)) / 10)
print(ccas.fit((x0, x4)).score((x0, x4)) / 10)

X = np.array([[mse1, cossim1, cka1, cca1], [mse2, cossim2, cka2, cca2], [mse3, cossim3, cka3, cca3], [mse4, cossim4, cka4, cca4]])
clustering = AffinityPropagation(damping=0.7, max_iter=1000).fit(X)
print(clustering.labels_)
print(clustering.predict([[mse0, cossim0, cka0, cca0]]))

weights_list = [model1.get_weights(), model2.get_weights(), model3.get_weights(), model4.get_weights()]
weights = model0.get_weights()
for i in range(len(weights)):
    for w in weights_list:
        weights[i] = weights[i] + w[i]
    weights[i] = weights[i] / (len(weights_list) + 1)
print(len(weights))
