import keras.layers as layers
import keras.models as models
import numpy
from keras.optimizer_v2.adam import Adam
from sklearn.preprocessing import StandardScaler
from ML01_DataSource import getData

# files = ["data_elephants_zebras_90.txt", "data_elephants_zebras_10.txt"]
# files = ["data_animals_150.txt", "data_animals_150.txt"]
files = ["data_elephants_rhinos_1000.txt", "data_elephants_rhinos_100.txt"]

# Использование стандартной библиотеки scikit-learn (Не надо изобретать велсипед)

animals, y_train, x_train, classes = getData(files[0])

# Для этого алгоритма шкалирование обязательно
scaler = StandardScaler().fit(x_train)
x_train = numpy.array(scaler.transform(x_train))
# Для нейронных сетей у (целевой вектор) должный быть числовым
# Это называется проблемой кодировки катогиральных признаков
y_train = numpy.array([ 0 if y == "носорог" else 1  for y in y_train])
# print(y_train)

# Простая однонейронная сеть (есть мнение, что это эквивалент логистической регрессия)
# model = models.Sequential()
# model.add(layers.Dense(1, activation="sigmoid", input_dim = 2))

# Выстроим более cложные сети (как упражнение)
model = models.Sequential()
model.add(layers.Dense(8, activation="relu", input_dim = 2)) # а почему бы не 8?
model.add(layers.Dense(2, activation="relu"))
model.add(layers.Dense(1, activation="sigmoid")) # 1 - птому что послдений, а классификация бинарная


model.compile(
    loss="binary_crossentropy",
    metrics=['accuracy']
)

model.fit(x_train, y_train, epochs=1000, batch_size=100)

# Проверим тчоность на тестовой выборке
animals, y_test, x_test, classes = getData(files[1])

x_test = numpy.array(scaler.transform(x_test))
# Для нейронных сетей у (целевой вектор) должный быть числовым
# Это называется проблемой кодировки катогиральных признаков
y_test = numpy.array([ 0 if y == "носорог" else 1  for y in y_test])

# К этому моменту уже обучена
model.evaluate(x_test, y_test)