import tensorflow.python.keras.layers as layers
import numpy

# Изучаем поведение нейрона

# Генерируем тестовые входные данные
data = range(-99, 100)
data = [ [d / 10] for d in data] # это называется unsqueeze
data = numpy.array(data) # keras ждет numpy массив
print(data)

# layer1 = layers.Dense(1, activation="linear", input_dim=1)
# layer1 = layers.Dense(1, activation="sigmoid", input_dim=1)
layer1 = layers.Dense(1, activation="relu", input_dim=1)

results = layer1(data) #  при первом вызове веса случайны
# Задаим предсказуемые веса
layer1.set_weights([numpy.array([[1.0]]),numpy.array([0.0])])
results = layer1(data)
print(results)

import matplotlib.pyplot as plt
x = [d[0] for d in data] # это называется squeeze
plt.plot(x, results)
plt.show()