from ML01_DataSource import getData

# Демонстрация алгоритма KNN

animals, labels, features, classes = getData("data_elephants_rhinos_100.txt")

# Гиперпрарметр модели (=число соседей для опредления)
k = 5

# Большинство алгоритмов МО требуют масшитабирования (scaling) входных значения
scale_age = 100
scale_weight = 2700
features = [ [f[0] / scale_age, f[1] / scale_weight ] for f in features ]
# print(features)

# Реализуем KNN
def predict(age, weight):
    distances = []
    # Определяем расстояние до каждой из точек обучающей выборки
    for i in range(0, len(features)):
        x = features[i][0]
        y = features[i][1]
        # Манхэттенская метрика
        dist = abs(age - x) + abs(weight - y)
        distances.append([dist, labels[i]])
    # Найдем k ближайших точек
    # print(distances)
    distances.sort(key=lambda d: d[0])
    distances = distances[0:k]
    # print(distances)
    # Подсчитаем число элементов класса 0 (носорогов) среди этих k
    n0 = len(list(filter(lambda d: d[1] == classes[0], distances)))
    # print(n0)
    if n0 > k / 2:
        return classes[0]
    else:
        return classes[1]

if __name__ == "__main__":
    print(predict(50 / scale_age, 1200 / scale_weight))
    print(predict(90 / scale_age, 2500 / scale_weight))
    print(predict(80 / scale_age, 100 / scale_weight))