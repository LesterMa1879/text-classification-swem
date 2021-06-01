import numpy as np
from test.Sogou_Classification.load_data import preprocess, load_datasets, read_vectors

from word2vec import Word2vec
from swem import SWEM
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import classification_report, confusion_matrix

word2vec = Word2vec("model/sgns.sogou.word", 0, 100)
X_train, y_train, X_test, y_test = word2vec.vectorization()
swem = SWEM(0)

X_train_data = swem.get_swem(X_train)
X_test_data = swem.get_swem(X_test)

# Logistic Regression
clf = LogisticRegression()
clf.fit(X_train_data, y_train)
# text_clf_lr.predict(X_new_data)
predicted_lr = clf.predict(X_test_data)
print(classification_report(predicted_lr, y_test))
# confusion_matrix(predicted_lr, y_test)

# SVM
clf = SGDClassifier(loss='hinge', penalty='l2')
clf.fit(X_train_data, y_train)
# text_clf_svm.predict(X_new_data)
predicted_svm = clf.predict(X_test_data)
print(classification_report(predicted_svm, y_test))
# confusion_matrix(predicted_svm, y_test)
