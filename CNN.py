from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.keras as ks

(train_X, train_y), (test_X, test_y) = mnist.load_data()

print(train_X[1].shape)
plt.imshow(train_X[1], cmap = "Greys")
plt.show()

CNN = ks.models.Sequential()

CNN.add(ks.layers.InputLayer((28, 28), 28))
CNN.add(ks.layers.Dense(15, "relu"))
CNN.add(ks.layers.Dense(10))

CNN.compile(optimizer = "Adam", loss = ks.losses.MeanSquaredError())

CNN.fit(train_X, train_y, batch_size = 28)

loss = CNN.evaluate(test_X, test_y, batch_size = 28)

print(loss)