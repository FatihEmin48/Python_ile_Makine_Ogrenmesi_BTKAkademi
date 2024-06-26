# -*- coding: utf-8 -*-
"""
Created on Mon May 13 11:36:45 2024

@author: fatih
"""

#Tahmin etmek istediğin veri sürekliyse: Gaussian Naive Bayes

#Veri nominal ise: Multinominal Naive Bayes

#2 tane nominal ise(evet-hayır): Bernoulli Naive Bayes

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd



veriler = pd.read_csv('veriler.csv')


x = veriler.iloc[:,1:4].values #bağımsız değişkenler
y = veriler.iloc[:,4:].values #bağımlı değişken



from sklearn.model_selection import train_test_split
x_train, x_test,y_train,y_test = train_test_split(x,y,test_size=0.33, random_state=0)



from sklearn.preprocessing import StandardScaler
sc=StandardScaler()

X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)



from sklearn.linear_model import LogisticRegression
logr = LogisticRegression(random_state=0)
logr.fit(X_train,y_train)

y_pred = logr.predict(X_test)
print(y_pred)
print(y_test)



from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
print(cm)



from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5, metric='minkowski')
knn.fit(X_train,y_train)

y_pred = knn.predict(X_test)
cm = confusion_matrix(y_test,y_pred)
print(cm)


from sklearn.svm import SVC
svc = SVC(kernel='linear')
svc.fit(X_train, y_train)

y_pred = svc.predict(X_test)

cm = confusion_matrix(y_test,y_pred)
print(cm)


from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train, y_train)

y_pred = gnb.predict(X_test)

cm = confusion_matrix(y_test,y_pred)
print(cm)