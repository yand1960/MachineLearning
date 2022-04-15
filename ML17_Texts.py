import nltk
from nltk.corpus import stopwords
import sklearn
import re

with(open("data/Exodus.txt", encoding="utf-8")) as f:
    text = f.read()

# Удаление отдельных символов
text = text.lower()
text = re.sub("[\.:\n=,;[\]]"," ", text)
text = re.sub("[0-9]","",text)
text = re.sub("\s+"," ", text)
print(text[0:1000])
words = text.split(" ")

# Удаленик стоп слов
# nltk.download('stopwords')
stop_words = stopwords.words('russian')
# print(stop_words)
words = [w for w in words if not w in stop_words ]
# print(words[0:500])

# Стеммизация (приведение слов к основам)
stemmer = nltk.stem.snowball.SnowballStemmer('russian')
words = [stemmer.stem(w) for w in words]
print(words[0:500])

# WordBag
cv = sklearn.feature_extraction.text.CountVectorizer()
bag_of_words = cv.fit_transform(words)
print(cv.get_feature_names())
# for v in bag_of_words.toarray():
#     print(len(v), v)

print("И пиши дальше алгоритмы ML")

