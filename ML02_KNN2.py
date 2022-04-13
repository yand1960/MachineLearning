# Реализуем алгоритм KNN в стиле, принятом в билиботеках MO

# Делаем класс модели KNN
class KNN:
    # Гиперпараметры модели передаются через конструктор модели
    def __init__(self, k, metrics = "manhattan"):
        self.k = k
        self.metrics = metrics

    # Метод, реализующий обучение (fit - подгонка)
    # В KNN очень просто: надо запомнить обучающую выборку
    # В общем случае, здесь может быть очень сложный код, вычисляюший параметры модели
    def fit(self, features, labels):
        self.features = features
        self.labels = labels
        self.classes = list(set(labels)).sort()


    # Реализуем KNN
    def predictOne(self, age, weight):
        distances = []
        # Определяем расстояние до каждой из точек обучающей выборки
        for i in range(0, len(self.features)):
            x = self.features[i][0]
            y = self.features[i][1]
            if self.metrics == "manhattan":
                # Манхэттенская метрика
                dist = abs(age - x) + abs(weight - y)
            else:
                # Эвклидова метрика
                dist = ((age - x) ** 2 + (weight - y) ** 2) ** 0.5
            distances.append([dist, labels[i]])
        # Найдем k ближайших точек
        # print(distances)
        distances.sort(key=lambda d: d[0])
        distances = distances[0:self.k]
        # print(distances)
        # Подсчитаем число элементов класса 0 среди этих k
        # n0 = len(list(filter(lambda d: d[1] == classes[0], distances)))
        # # Надо бы переписать для n>2 классификаторов:
        # if n0 > self.k / 2:
        #     return classes[0]
        # else:
        #     return classes[1]
        # Подсчет для n>2 классификаторов в духе спортивной краткости:
        return max([[len(list(filter(lambda d: d[1] == c, distances))), c] for c in classes])[1]

    def predict(self, data):
        results = []
        for item in data:
            results.append(self.predictOne(item[0], item[1]))
        return results

if __name__ == "__main__":
    from ML01_DataSource import getData

    # Откуда-то берут входные данные для обучения
    animals, labels, features, classes = getData("data_elephants_rhinos_100.txt")

    # Большинство алгоритмов МО требуют масшитабирования (scaling) входных значений
    # В общем случае, здесь может быть сложный препроцессинг данных
    scale_age = 100
    scale_weight = 2700
    features = [[f[0] / scale_age, f[1] / scale_weight] for f in features]

    # Создание модели
    model = KNN(5)
    # Обучение модели
    model.fit(features, labels)

    # Предварительная проверка обученной модели
    print(model.predictOne(50 / scale_age, 1200 / scale_weight))
    print(model.predictOne(90 / scale_age, 2500 / scale_weight))
    print(model.predictOne(80 / scale_age, 100 / scale_weight))
    print(model.predict([
        [50 / scale_age, 1200 / scale_weight],
        [90 / scale_age, 2500 / scale_weight],
        [80 / scale_age, 100 / scale_weight]
    ]))

    # Проверить точность предсказаний модели на тестовой выборке (testing set)
    # 1. Не вполне корректный вариант с использованием обучающей выборки в качестве тестовой

    predictions = model.predict(features)
    errors = 0
    for i in range(0, len(labels)):
        if predictions[i] != labels[i]:
            errors += 1
    print(f"Точность на обучающей выборке: {1 - errors / len(predictions)}")

    # 2. использовать для обучения входную выборку за вычетом некоей случайно подвыборки.
    # Случайную подвыборку использовать в качестве тестовой - не делаем

    # 3. Получаем тестовую выборку от доброго дяди
    # Откуда-то берут входные данные
    animals, labels_test, features_test, classes = getData("data_elephants_rhinos_1000.txt")

    # Масштабируем так же, как обучающую
    scale_age = 100
    scale_weight = 2700
    features_test = [[f[0] / scale_age, f[1] / scale_weight] for f in features_test]
    predictions = model.predict(features_test)
    errors = 0

    for i in range(0, len(labels_test)):
        if predictions[i] != labels_test[i]:
            errors += 1
    print(f"Точность на тестовой выборке: {1 - errors / len(predictions)}")
