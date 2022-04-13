# Задача клатеризации - разделить на группы
# Иногда это называется обучением без учителя

# Ипользование алгоритма Логистической регрессии.
# На самом деле, это не регрессия, к классификатор, а название - историческое

from sklearn.cluster import KMeans
from ML01_DataSource import getData
from sklearn.preprocessing import StandardScaler

files = ["data_elephants_zebras_90.txt"]
# files = ["data_animals_150.txt", "data_animals_150.txt"]
# files = ["data_elephants_rhinos_1000.txt", "data_elephants_rhinos_100.txt"]

animals, labels, features, classes = getData(files[0])

# Для этого алгоритма шкалирование обязательно
scaler = StandardScaler().fit(features)
features = scaler.transform(features)


# Создаем и обучаем модель
model = KMeans(n_clusters=2)
model.fit(features)

predictions = model.predict(features)
print(predictions)

# Визуализация результатов
import matplotlib.pyplot as plt

for i in range(0,len(features)):
    x = features[i][0]
    y = features[i][1]
    if predictions[i] == 0:
        plt.plot(x, y, "or")
    else:
        plt.plot(x, y, "ob")
plt.show()

# Попробуйте кластеризацивать ирисы
