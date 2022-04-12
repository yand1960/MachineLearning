from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from numpy import ndarray
from ML01_DataSource import getData

# Использование стандртной библиотеки scilearn (Не надо изобретать велсипед)

animals, labels, features, classes = getData("data_elephants_rhinos_100.txt")

# Масшатбируем стандартным скейлером
scaler = MinMaxScaler()
scaler.fit(features)
features: ndarray = scaler.transform(features)

# print(features)

# Создаем и обучаем модель
model = KNeighborsClassifier(5)
model.fit(features,labels)

# Проверяем точность на тестовой выборке
predictions = model.predict(features)

errors = 0
for i in range(0, len(labels)):
    if predictions[i] != labels[i]:
        errors += 1
print(f"Точность на обучающей выборке: {1 - errors / len(predictions)}")

animals, labels_test, features_test, classes = getData("data_elephants_rhinos_1000.txt")
# Масштабируем тестовую так же, как обучающую (тем же скейлером)
features_test = scaler.transform(features_test)

predictions = model.predict(features_test)
errors = 0

for i in range(0, len(labels_test)):
    if predictions[i] != labels_test[i]:
        errors += 1
print(f"Точность на тестовой выборке: {1 - errors / len(predictions)}")

# ЗАДАЧИ
# 1. Поисcледуйте слонов и зебр
# 2. Надо бы переписать код для n>2 классификаторов в файле KNN2
