# Задача клаcтеризации - разделить выборку на группы
# Иногда тако обучение называется обучением без учителя

from sklearn.cluster import KMeans, SpectralClustering, DBSCAN, OPTICS, Birch
from ML01_DataSource import getData
from sklearn.preprocessing import StandardScaler

files = ["data_elephants_zebras_90.txt"]
# files = ["data_animals_150.txt", "data_animals_150.txt"]
# files = ["data_elephants_rhinos_1000.txt", "data_elephants_rhinos_100.txt"]

animals, labels, features, classes = getData(files[0])

# Для этого алгоритма шкалирование обязательно
scaler = StandardScaler().fit(features)
features = scaler.transform(features)

# Создаем модель
model = KMeans(n_clusters=2)
# model = SpectralClustering(n_clusters=2)
# model = Birch(n_clusters=2)

predictions = model.fit_predict(features)
print(predictions) #Видно, что кластеризация прошла не по признаку слонов и зебр

# Визуализация результатов (чтобы убедиться)
import matplotlib.pyplot as plt

colors = "rbgykmcrbgykmc"
for i in range(0,len(features)):
    x = features[i][0]
    y = features[i][1]
    plt.plot(x, y, f"o{colors[predictions[i]]}")

plt.show()

# Попробуйте кластеризацивать ирисы
