import tensorflow as tf
import numpy as np
from numpy.random import seed
import sklearn.preprocessing as skp

seed(1)
tf.random.set_seed(1)

train_x_location = "x_train16.csv"
train_y_location = "y_train16.csv"
test_x_location = "x_test.csv"
test_y_location = "y_test.csv"

print("Reading training data")
x_train = np.loadtxt(train_x_location, dtype="float_", delimiter=",")
y_train = np.loadtxt(train_y_location, dtype="float_", delimiter=",")

m, n = x_train.shape
m_labels, = y_train.shape
l_min = y_train.min()

assert m_labels == m, "x_train and y_train should have same length."
assert l_min == 0, "each label should be in the range 0 - k-1."
k = y_train.max() + 1

print(m, "examples ,", n, "features ,", k, "categiries.")

scaler = skp.MinMaxScaler()
x_train = scaler.fit_transform(x_train)
model = tf.keras.models.Sequential([
    tf.keras.Input((n,)),
    tf.keras.layers.Dense(140, activation=tf.keras.activations.sigmoid),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(400, activation=tf.keras.activations.sigmoid),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(140, activation=tf.keras.activations.sigmoid),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(k, activation=tf.nn.softmax)
])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

print("train")
model.fit(x_train, y_train, epochs=1000, batch_size=32)

print("Reading testing data")
x_test = np.loadtxt(test_x_location, dtype="float_", delimiter=",")
y_test = np.loadtxt(test_y_location, dtype="float_", delimiter=",")

m_test, n_test = x_test.shape
m_test_labels, = y_test.shape
l_min = y_train.min()

assert m_test_labels == m_test, "x_test and y_test should have same length."
assert n_test == n, "train and x_test should have same number of features."
print(m_test, "test examples.")

x_test = scaler.transform(x_test)
print("evaluate")
model.evaluate(x_test, y_test)