import keras.layers as layers
import keras.models as models
from keras.optimizer_v2.adam import Adam
from sklearn.preprocessing import MinMaxScaler
import math

# Предсказываем значения в числовом ряду

# Генерируем тестовые входные данные
data = range(0, 100)
x = [ [d / 10] for d in data]
# y = [ [d ** 2] for d in data ] # x квадрат
y = [ [math.exp(d)] for d in data ] # "взрывная" экспонента

# Шкалируем входные данные
scalerX = MinMaxScaler().fit(x)
scalerY = MinMaxScaler().fit(y)
x = scalerX.transform(x)
y = scalerY.transform(y)
print(x)
print(y)

# Однослойная однонейронная модель не способна выдать что-то иное,
# кроме линейной функции
# model = models.Sequential()
# model.add(layers.Dense(1,activation="linear", input_dim=1))

# Делаем многослойную сеть для апрокисмации квадратов
# Хороший результат: 100 нейронов, learning_rate = 0.01, 1000 эпох
model = models.Sequential()
model.add(layers.Dense(100,activation="relu", input_dim=1))
model.add(layers.Dense(1, activation="linear"))

model.compile(
    optimizer= Adam(learning_rate=0.001 ),
    loss = "mean_squared_error"
)

model.fit(x,y, epochs=10000, verbose=1, batch_size=100)
predictions = model.predict(x)

# Делаем многослойную сеть для апрокисамци "резкой" экспоненты
# Хороший результат: 100 нейронов, learning_rate = 0.01, 1000 эпох
# Решающий параметр при этом: batch_size = 100
# Отличный результат: 100 нейронов, learning_rate = 0.01, 5000 эпох
# Экспериментолаьно: кастомная функция потерь,
# model = models.Sequential()
# model.add(layers.Dense(100, activation="relu", input_dim=1))
# model.add(layers.Dense(1, activation="linear"))
#
# import keras.backend as K
# def max_abs_error(y_true, y_pred):
#     print(y_true)
#     return K.max(K.abs(y_true - y_pred))
#
# model.compile(
#     optimizer=Adam(learning_rate=0.01),
#     loss = "mse"
#     # loss=max_abs_error
# )
#
# model.fit(x,y, epochs=5000, verbose=0, batch_size=100)
# predictions = model.predict(x)

# Визуализация результатов
import matplotlib.pyplot as plt
plt.plot(x, y)
plt.plot(x, predictions)

plt.show()