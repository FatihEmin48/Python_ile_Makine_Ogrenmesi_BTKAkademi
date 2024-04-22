# -*- coding: utf-8 -*-
"""
Created on Sun Mar 10 14:53:41 2024

@author: fatih
"""

#Kütüphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Veri yukleme
veriler = pd.read_csv('maaslar.csv')

#Data frame dilimleme (Slice)
x = veriler.iloc[:,1:2]
y = veriler.iloc[:,2:]

#NumPY Dizi (Array) Dönüşümü
X = x.values
Y = y.values


#Doğrusal Model Oluşturma (Linear Regression)
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,Y)




#Doğrusal Olmayan (Nonlinear Model) Oluşturma (Polynomial Regression)
#2.Derecede Polinom
from sklearn.preprocessing import PolynomialFeatures
poly_reg2 = PolynomialFeatures(degree = 2)
x_poly2 = poly_reg2.fit_transform(X)
lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly2,y)


#4.Derecede Polinom
poly_reg3 = PolynomialFeatures(degree = 4)
x_poly3 = poly_reg3.fit_transform(X)
lin_reg3 = LinearRegression()
lin_reg3.fit(x_poly3,y)



#Görseleştirme
plt.scatter(X,Y,color='red')
plt.plot(x,lin_reg.predict(X), color = 'blue')
plt.show()


plt.scatter(X,Y,color = 'red')
plt.plot(X,lin_reg2.predict(poly_reg2.fit_transform(X)), color = 'blue')
plt.show()


plt.scatter(X,Y,color = 'red')
plt.plot(X,lin_reg3.predict(poly_reg3.fit_transform(X)), color = 'blue')
plt.show()


#Tahminler
print(lin_reg.predict([[11]]))
print(lin_reg.predict([[6.6]]))

print(lin_reg2.predict(poly_reg2.fit_transform([[6.6]])))
print(lin_reg2.predict(poly_reg2.fit_transform([[11]])))