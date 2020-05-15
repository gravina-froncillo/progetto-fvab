import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import sklearn
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
from sklearn.svm import LinearSVC

dataset = pd.read_csv('DatasetCelebA-5C_4S_inv.csv', sep=';')
X = dataset.iloc[:, 0:80].values
print(X.shape)
y = dataset.iloc[:, 81].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0 )
classifier = SVC()
history = classifier.fit(X_train, y_train)
predicted = classifier.predict(X_test)
print("Classification report for classifier %s:\n%s\n"
      % (classifier, classification_report(y_test, predicted)))
disp = sklearn.metrics.plot_confusion_matrix(classifier, X_test, y_test)
disp.figure_.suptitle("Confusion Matrix")
print("Confusion matrix:\n%s" % disp.confusion_matrix)
print("Accuracy Score:")
print(accuracy_score(y_test, predicted))
