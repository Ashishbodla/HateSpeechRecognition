import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
import numpy as np

train = pd.read_csv(r"CountVectorised_train.csv")
test = pd.read_csv(r"CountVectorised_test.csv")

train.dropna(inplace=True)
test.dropna(inplace=True)

X_train = train.drop(columns = ['class','Hate','Offensive','Neither'])
y_trian = train['class']

X_test = test.drop(columns = ['class','Hate','Offensive','Neither'])
y_test = test['class']

from sklearn import svm

clf = svm.SVC()
clf.fit(X_train, y_trian)
y_Pred_svm = clf.predict(X_test)
C_svm = confusion_matrix(y_test,y_Pred_svm)
print(np.trace(np.matrix(C_svm)))
