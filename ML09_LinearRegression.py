# Задача реагресии - экстраполировать или интерполировать числовые данные
# Например, сделать модель, которая предсказывает вес слоан по его возрасту


from sklearn.linear_model import LinearRegression
from ML01_DataSource import getData
from sklearn.preprocessing import StandardScaler

# files = ["data_elephants_zebras_90.txt", "data_elephants_zebras_10.txt"]
# files = ["data_animals_150.txt", "data_animals_150.txt"]
files = ["data_elephants_rhinos_1000.txt"]


animals, labels, features, classes = getData(files[0])
# Оставляем только данные о слонах
features = features[0:500]

x = [[d[0]] for d in features]
y = [[d[1]] for d in features]
# print(x)
# print(y)


# Создаем и обучаем модель
model = LinearRegression()
model.fit(x,y)

# Получаем предсказание для каких-то данных
prediction = model.predict([[50]])
print(prediction)
