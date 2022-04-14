import keras.layers as layers
import keras.models as models
import numpy
from keras.optimizer_v2.adam import Adam
from sklearn.preprocessing import StandardScaler
from ML01_DataSource import getData

# files = ["data_elephants_zebras_90.txt", "data_elephants_zebras_10.txt"]
# files = ["data_animals_150.txt", "data_animals_150.txt"]
files = ["data_animals_150.txt", "data_animals_150.txt"]

# Использование стандартной библиотеки scikit-learn (Не надо изобретать велсипед)

animals, y_train, x_train, classes = getData(files[0])

# Для этого алгоритма шкалирование обязательно
scaler = StandardScaler().fit(x_train)
x_train = numpy.array(scaler.transform(x_train))
# Для нейронных сетей у (целевой вектор) должный быть числовым
# Это называется проблемой кодировки категориальных признаков
# Релизуем one-hot кодировку
y_train = numpy.array(
    [
      [1 if y == "зебра" else 0 ,
       1 if y == "носорог" else 0,
       1 if y == "слон" else 0]
        for y in y_train
    ])
# print(y_train)

# Простая сеть
model = models.Sequential()
model.add(layers.Dense(3, activation="sigmoid", input_dim = 2))

# Выстроим более cложные сети (как упражнение)
# model = models.Sequential()
# model.add(layers.Dense(8, activation="relu", input_dim = 2))
# model.add(layers.Dense(3, activation="softmax"))


model.compile(
    loss="categorical_crossentropy",
    metrics=['accuracy']
)

model.fit(x_train, y_train, epochs=3000)
y_predict: list = model.predict(x_train)
print(y_predict)

# Ошибки отдельно по классам
# errors = [0,0,0]
# for i in range(0, len(x_train)):
#     predictedClass = list(y_predict[i]).index(max(y_predict[i]))
#     realClass = list(y_train[i]).index(max(y_train[i]))
#     if realClass != predictedClass:
#         errors[realClass] += 1
# print(errors)

# model.evaluate(x_train,y_train)

# 1. Попытайтесь решить проблему "резкой" экспоненты (в файле 12)
# https://stackoverflow.com/questions/58163571/keras-python-custom-loss-function-to-give-the-maximum-value-of-absolute-differe
# 2. Добавьте слои в эту модель. Не улучшится ли точность?