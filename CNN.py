from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.keras as ks

(train_X, train_y), (test_X, test_y) = mnist.load_data()

print(train_X[1].shape)
plt.imshow(train_X[1], cmap = "Greys")
#plt.show()

CNN = ks.models.Sequential()

# convolutional layer
CNN.add(ks.layers.Conv2D(25, kernel_size=(3,3), strides=(1,1), padding='valid', activation='relu', input_shape=(32, 28,28,1)))
CNN.add(ks.layers.MaxPool2D(pool_size=(1,1)))
# flatten output of conv
CNN.add(ks.layers.Flatten())
# hidden layer
CNN.add(ks.layers.Dense(100, activation='relu'))
# output layer
CNN.add(ks.layers.Dense(10, activation='softmax'))


CNN.compile(optimizer = "adam", loss = ks.losses.MeanSquaredError())

CNN.fit(train_X, train_y, batch_size = 32, validation_data = (test_X, test_y))

loss = CNN.evaluate(test_X, test_y, batch_size = 32)

print(loss)