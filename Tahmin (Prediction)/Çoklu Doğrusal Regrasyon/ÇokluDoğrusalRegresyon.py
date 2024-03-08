# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 09:33:04 2024

@author: fatih
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


veriler = pd.read_csv('veriler.csv')
print(veriler)

#Eksik verileri düzeltme
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

Yas = veriler.iloc[:,1:4].values
print(Yas)

imputer = imputer.fit(Yas[:,1:4])
Yas[:,1:4] = imputer.transform(Yas[:,1:4])
print(Yas)



ulke = veriler.iloc[:,0:1].values
print(ulke)

from sklearn import preprocessing

le = preprocessing.LabelEncoder()
ulke[:,0] = le.fit_transform(veriler.iloc[:,0])

ohe = preprocessing.OneHotEncoder()
ulke = ohe.fit_transform(ulke).toarray()
print(ulke)



c = veriler.iloc[:,-1:].values
print(c)

from sklearn import preprocessing

le = preprocessing.LabelEncoder()

c[:,-1] = le.fit_transform(veriler.iloc[:,-1])

ohe = preprocessing.OneHotEncoder()
c = ohe.fit_transform(c).toarray()
print(c)


#Numpy dizileri dataframe dönüşümü
#Verilerin birleştirilmesi
sonuc = pd.DataFrame(data=ulke, index = range(22), columns = ['fr','tr','us'])
print(sonuc)

sonuc2 = pd.DataFrame(data=Yas, index = range(22), columns = ['boy','kilo','yas'])
print(sonuc2)

cinsiyet = veriler.iloc[:,-1].values
print(cinsiyet)

sonuc3 = pd.DataFrame(data = c[:,:1], index = range(22), columns = ['cinsiyet'])
print(sonuc3)


#Dataframe birleştirme işlemi
s = pd.concat([sonuc,sonuc2], axis=1)
print(s)

s2 = pd.concat([s,sonuc3], axis=1)
print(s2)


#Veri Kümesinin Eğitim ve Test Olarak Bölünmesi
from sklearn.model_selection import train_test_split

x_train, x_test,y_train,y_test = train_test_split(s,sonuc3,test_size=0.33, random_state=0)


#Verilerin ölçeklendirilmesi
#Öznitelik Ölçekleme
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)



#Çoklu Değişken Linear Model Uluşturulması
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)

y_pred = regressor.predict(x_test)

boy = s2.iloc[:,3:4].values
print(boy)

sol = s2.iloc[:,:3]
sag = s2.iloc[:,4:]

veri = pd.concat([sol,sag], axis=1)

x_train, x_test,y_train,y_test = train_test_split(veri,boy,test_size=0.33, random_state=0)

r2 = LinearRegression()
r2.fit(x_train, y_train)

y_pred = regressor.predict(x_test)

