# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 11:40:46 2024

@author: fatih
"""

#Kütüphaneler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#Veri Ön işleme

#veri yükleme
veriler = pd.read_csv('satislar.csv')
print(veriler)

#Veriyi ayırma
aylar = veriler[['Aylar']]
satislar = veriler[['Satislar']]



#Veri Kümesinin Eğitim ve Test Olarak Bölünmesi
from sklearn.model_selection import train_test_split

x_train, x_test,y_train,y_test = train_test_split(aylar,satislar,test_size=0.33, random_state=0)



'''
#Verilerin ölçeklendirilmesi
#Öznitelik Ölçekleme
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)

Y_train = sc.fit_transform(y_train)
Y_test = sc.fit_transform(y_test)
'''


#Basit Doğrusal Regrasyon
#Model İnşası(Linear Regrasyon)
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(x_train, y_train)

tahmin = lr.predict(x_test)


#verilerin sıralanması
x_train = x_train.sort_index()
y_train = y_train.sort_index()

#Grafik çizdirme
plt.plot(x_train, y_train)
plt.plot(x_test, lr.predict(x_test))


plt.title("Aylara Göre Satış")
plt.xlabel("Aylar")
plt.ylabel("Satışlar")


