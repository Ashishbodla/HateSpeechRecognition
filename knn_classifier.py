import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix 
import numpy as np


train = pd.read_csv(r"CountVectorised_train.csv")
test = pd.read_csv(r"CountVectorised_test.csv")

train.dropna(inplace=True)
test.dropna(inplace=True)

X_train = train.drop(columns = ['class','Hate','Offensive','Neither'])
y_train = train['class']
X_test = test.drop(columns = ['class','Hate','Offensive','Neither'])
y_test = test['class']

# Model building for each number of neighbors

for i in range(3,20):
    neigh = KNeighborsClassifier(n_neighbors=i)
    neigh.fit(X_train, y_train)

    y_pred = neigh.predict(X_test)
    C = confusion_matrix(y_test,y_pred)
    print(i,np.trace(np.matrix(C))) 
