import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
import numpy as np
import pickle
from sklearn import svm

train = pd.read_csv("CountVectorised_train.csv")
test = pd.read_csv("CountVectorised_test.csv")

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
    pickle.dump(neigh,r'.\Count_Vectors_models\Neigh_{}.sav'.format(i))



clf = svm.SVC()
clf.fit(X_train, y_trian)
y_Pred_svm = clf.predict(X_test)
C_svm = confusion_matrix(y_test,y_Pred_svm)
print(np.trace(np.matrix(C_svm)))
pickle.dump(clf,'SVM_SVC.sav')

MNB= MultinomialNB().fit(X_train, train['class'])
MNB_preds = MNB.predict(X_test)
print(confusion_matrix(test['class'],MNB_preds))
print (classification_report(test['class'], MNB_preds))
pickle.dump(MNB, 'MNB_counts.sav')

for N in range(100,1000,100):
    rfc = RandomForestClassifier(n_estimators=N,min_samples_split=5,random_state=42,class_weight="balanced")
    rfc.fit(X_train, y_trian)
    rfc_preds = rfc.predict(X_test)
    C_rfc = confusion_matrix(y_test,rfc_preds)
    print(N,np.trace(np.matrix(C_rfc)))
    pickle.dump(rfc, 'RFC_{}.sav'.format(N))
