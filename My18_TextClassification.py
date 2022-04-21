import nltk
import re
from sklearn.svm import SVC

# Классификация текстов.
# Два текста (Исход и Матфей) нормализуются,
# а затем режутся на фрагменты размером по FRAGMENT_SIZE слов.
# Для каждого фрагмента определяется вектор частот вхождения слов общего словаря,
# вектор используется в качестве точки признаков для этого фрагмента.
# Далее применяется SVM. Результат на удивление точен.


def normalize_file(file):
    with(open(file, encoding="utf-8")) as f:
        text = f.read()

    # Удаление отдельных символов
    text = text.lower()
    text = re.sub("[\.:\n=,;[\]\!\?\"\-]"," ", text)
    text = re.sub("[0-9]","",text)
    text = re.sub("\s+"," ", text)
    print(text[0:1000])

    # Разделение текста на слова называется "токенизация"
    words = text.split(" ")

    # Удаление стоп слов
    # nltk.download('stopwords')  # надо вызвать однократно
    stop_words = nltk.corpus.stopwords.words('russian')
    words = [w for w in words if not w in stop_words ]

    # Стемизация (приведение слов к основам)
    stemmer = nltk.stem.snowball.SnowballStemmer('russian')
    words = [stemmer.stem(w) for w in words]

    return words


# Формирование тестовой и обучающей выборок
FILES = ['data/Exodus.txt', 'data/Mathew.txt']
FRAGMENT_SIZE = 100 # при значении 5 точность получается 87%, при 60 и более - 100%
TRAIN_TO_TEST_RATIO = 10 # отношение размеров обучающей и тестовой выборок

texts = []
tokens = []
train_x = []
train_y = []
test_x = []
test_y = []

for file in FILES:
    words = normalize_file(file)
    texts.append(words)
    tokens += words
tokens  = sorted(list(set(tokens))) # Фактически, это общий словарь всех текстов
print([len(t) for t in texts])
print(len(tokens), tokens)

for text, file in zip(texts, FILES):
    for i in range(0 , len(text) // FRAGMENT_SIZE):
        fragment: list = text[i * FRAGMENT_SIZE :  (i + 1) * FRAGMENT_SIZE]
        label = file
        feature = []
        # Определение частоты вхождения каждого токена в данный фрагмент
        feature = [fragment.count(t) for t in tokens]

        if i % TRAIN_TO_TEST_RATIO != 0:
            train_x.append(feature)
            train_y.append(label)
        else:
            test_x.append(feature)
            test_y.append(label)

# Ипользование алгоритма семейства SVM из стандартной библиотеки

# Создаем и обучаем модель
model = SVC(kernel="linear")
model.fit(train_x , train_y)

# Проверяем точность на обучающей и тестовой выборках
predictions = model.predict(train_x)
errors = 0
for i in range(0, len(train_y)):
    if predictions[i] != train_y[i]:
        errors += 1
print(f"Точность на обучающей выборке: {1 - errors / len(predictions)}")

predictions = model.predict(test_x)
errors = 0
for i in range(0, len(test_y)):
    if predictions[i] != test_y[i]:
        errors += 1
print(f"Точность на тестовой выборке: {1 - errors / len(predictions)}")
