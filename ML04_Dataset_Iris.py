# Наборы данных из стандартных библиотек (пример)

import sklearn.datasets as ds
from sklearn.neighbors import KNeighborsClassifier
import numpy
from numpy import ndarray
from sklearn.preprocessing import StandardScaler

irises = ds.load_iris()
# print(irises)
features: ndarray = irises.data
# print(features)
targets: ndarray = irises.target
# print(targets)

# Шкалирование
scaler = StandardScaler().fit(features)
features = scaler.transform(features)
print(features)

# Сгенерируем тестовую и обучающаю выборку
training_features = numpy.concatenate([features[0:35], features[50:85], features[100:135]])
training_targets = numpy.concatenate([targets[0:35], targets[50:85], targets[100:135]])
test_features = numpy.concatenate([features[35:50], features[85:100], features[135:150]])
test_targets = numpy.concatenate([targets[35:50], targets[85:100], targets[135:150]])

# Классифицируем
model = KNeighborsClassifier(5)
model.fit(training_features, training_targets)
predictions = model.predict(test_features)

errors = 0
for i in range(0, len(test_features)):
    if predictions[i] != test_targets[i]:
        errors += 1
print(f"Точность: {1 - errors / len(test_features) }")


