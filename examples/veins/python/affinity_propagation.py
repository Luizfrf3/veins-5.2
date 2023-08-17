import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, datasets
from sklearn.cluster import AffinityPropagation
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
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(num_classes, name="my_dense", activation="softmax")
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

mse1 = mse(w0, w1)
mse2 = mse(w0, w2)
mse3 = mse(w0, w3)
mse4 = mse(w0, w4)
print(mse1)
print(mse2)
print(mse3)
print(mse4)

cossim1 = cossim(w0, w1)
cossim2 = cossim(w0, w2)
cossim3 = cossim(w0, w3)
cossim4 = cossim(w0, w4)
print(cossim1)
print(cossim2)
print(cossim3)
print(cossim4)

(x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
x_train = x_train.astype("float32")[:20] / 255

inter_model0 = keras.Model(inputs=model0.input, outputs=model0.get_layer("my_dense").output)
inter_model1 = keras.Model(inputs=model1.input, outputs=model1.get_layer("my_dense").output)
inter_model2 = keras.Model(inputs=model2.input, outputs=model2.get_layer("my_dense").output)
inter_model3 = keras.Model(inputs=model3.input, outputs=model3.get_layer("my_dense").output)
inter_model4 = keras.Model(inputs=model4.input, outputs=model4.get_layer("my_dense").output)

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

x0 = np.array(inter_model0(x_train))
x1 = np.array(inter_model1(x_train))
x2 = np.array(inter_model2(x_train))
x3 = np.array(inter_model3(x_train))
x4 = np.array(inter_model4(x_train))

cka1 = cka(x0, x1)
cka2 = cka(x0, x2)
cka3 = cka(x0, x3)
cka4 = cka(x0, x4)
print(cka1)
print(cka2)
print(cka3)
print(cka4)

print(linear_CKA(x0, x1))
print(linear_CKA(x0, x2))
print(linear_CKA(x0, x3))
print(linear_CKA(x0, x4))

cca1 = cca(x0, x1)
cca2 = cca(x0, x2)
cca3 = cca(x0, x3)
cca4 = cca(x0, x4)
print(cca1)
print(cca2)
print(cca3)
print(cca4)

X = np.array([[mse1, cossim1, cka1, cca1], [mse2, cossim2, cka2, cca2], [mse3, cossim3, cka3, cca3], [mse4, cossim4, cka4, cca4]])
clustering = AffinityPropagation().fit(X)
print(clustering.labels_)

weights_list = [model1.get_weights(), model2.get_weights(), model3.get_weights(), model4.get_weights()]
weights = model0.get_weights()
for i in range(len(weights)):
    for w in weights_list:
        weights[i] = weights[i] + w[i]
    weights[i] = weights[i] / (len(weights_list) + 1)
print(weights)
