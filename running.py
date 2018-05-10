import re
import nltk
import glob
import errno
from sklearn import svm
from sklearn.feature_extraction.text import CountVectorizer

training_set = []
test_set= []
trainingSet_label=[]
testSet_label=[]
path = 'articlesCulture/*.txt'
files = glob.glob(path)
for name in files:
    Words=""
    try:
        with open(name ,encoding='utf-8') as f:
            text = f.read()
            for word in text:
                if ord(word)==10:
                    Words+=' '
                else:
                    Words+=word
    except IOError as exc:
        if exc.errno != errno.EISDIR:
            raise
    training_set.append(Words)
    trainingSet_label.append(["culture"])


new_path = 'articlesReligion/*.txt'
new_files = glob.glob(path)
for name in new_files:
    Words=""
    try:
        with open(name , encoding='utf-8') as f:
            text = f.read()
            for word in text:
                if ord(word)==10:
                    Words+=' '
                else:
                    Words+=word
    except IOError as exc:
        if exc.errno != errno.EISDIR:
            raise
    test_set.append(Words)
    testSet_label.append(["religion"])


data=[]
labels=[]
for i in range(2495):
    data.append(training_set[i])
    labels.append(trainingSet_label[i])
    data.append(test_set[i])
    labels.append(testSet_label[i])

count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(data[:3000])
X_train_counts.shape

tfidf_transformer = TfidfTransformer()
X_train = tfidf_transformer.fit_transform(X_train_counts)
X_train.shape

nb=GaussianNB()
nb.fit(X_train.toarray(),labels[:3000])
print (nb.predict(X_train.toarray()))
