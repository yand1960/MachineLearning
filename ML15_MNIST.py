from keras.datasets import mnist
import keras.models as models
import keras.layers as layers
import keras.initializers as ini
from keras.preprocessing import image
import matplotlib.pyplot as plt
import cv2
import numpy

# Получаем и иселдуем данные из MNIST
(train_x, train_y), (test_x, test_y) = mnist.load_data()
print(len(train_x))
train_x = train_x / 255
test_x = test_x / 255
picture = train_x[123]
# print(picture)
# print(train_y[123])
# plt.imshow(picture, cmap="Greys", interpolation='nearest')
# plt.show()
# exit()

# Получаем картинку из нашего файла
picture = cv2.imread("data/T1210.jpg")
picture = numpy.array(picture) / 255
# print(picture)
model = models.Sequential()
model.add(layers.Conv2D(input_shape=(12,10,3), kernel_size=(3,3), filters=3))
model.add(layers.MaxPool2D(2,strides=2))
result = model.layers[0](numpy.array([picture]))[0] #  unsqueese и squeeze

plt.imshow(result)
plt.show()
exit()

model = models.Sequential()
model.add(layers.Flatten(input_shape=(28,28)))
# layer = model.layers[0]
# result = layer(numpy.array([picture]))
# print(result[0])
model.add(layers.Dense(10, activation="relu")) # 100 - здравый смысл
model.add(layers.Dense(10, activation= "softmax"))

model.compile(
    loss = "sparse_categorical_crossentropy", # categorical_crossentropy - только для one-hot
    metrics=['accuracy']
)

model.fit(train_x, train_y, epochs=5)
model.evaluate(test_x, test_y)

