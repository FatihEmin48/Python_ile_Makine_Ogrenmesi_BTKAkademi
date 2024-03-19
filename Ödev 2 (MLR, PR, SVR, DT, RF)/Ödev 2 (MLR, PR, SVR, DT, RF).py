# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 11:49:42 2024

@author: fatih
"""

#Kütüphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import r2_score
import statsmodels.api as sm


#Veri yukleme
#Genelde ID kolonu alınmaz çünkü ezberlemeye neden olabilir
veriler = pd.read_csv('maaslar_yeni.csv')
#Elimizdeki veri için ünvan kukla değişken (dummy variable) çünkü ünvan seviyesi kolonumuz var

x = veriler.iloc[:,2:3]
y = veriler.iloc[:,5:]
X = x.values
Y = y.values
 

#Linear regression
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, Y)


#Linear regresyonda P value'leri hesaplama
model = sm.OLS(lin_reg.predict(X), X)
print(model.fit().summary())



#Polynomial regression
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 2)
x_poly = poly_reg.fit_transform(X)
print(x_poly)
lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly,y)


from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
x_poly = poly_reg.fit_transform(X)
print(x_poly)
lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly,y)


#Polinom regresyonda P value'leri hesaplama
print("poly ols")
model2 = sm.OLS(lin_reg2.predict(poly_reg.fit_transform(X)), X)
print(model2.fit().summary())



#Verilerin Ölçeklenmesi
from sklearn.preprocessing import StandardScaler

sc1=StandardScaler()
x_olcekli = sc1.fit_transform(X)

sc2=StandardScaler()
y_olcekli = np.ravel(sc2.fit_transform(Y.reshape(-1,1)))





#Support Vector Regression - SVR
from sklearn.svm import SVR

svr_reg = SVR(kernel='rbf')
svr_reg.fit(x_olcekli, y_olcekli)

#SVR'da P value'leri hesaplama
print("SVR poly")
model3 = sm.OLS(svr_reg.predict(x_olcekli), x_olcekli)
print(model3.fit().summary())




#Karar Ağacı Regresyonu
from sklearn.tree import DecisionTreeRegressor
r_dt = DecisionTreeRegressor(random_state=0)
r_dt.fit(X, Y)

Z = X + 0.5
K = X - 0.4


#Decisio Tree'de P value'leri hesaplama
print("Decision Tree poly")
model4 = sm.OLS(r_dt.predict(X), X)
print(model4.fit().summary())



#Random Forest
#Majority Vote => Çoğunluk ne derse ona karar veriyor
from sklearn.ensemble import RandomForestRegressor #Regressor tahmin için kullanılıyor
rf_reg = RandomForestRegressor(n_estimators = 10, random_state=0)
rf_reg.fit(X, Y.ravel())

print(rf_reg.predict([[6.6]]))


#Random Forest'da P value'leri hesaplama
print("Random Forest poly")
model5 = sm.OLS(rf_reg.predict(X), X)
print(model5.fit().summary())


#Algoritmaların Değerlendirilmesi
print('\n\n----------------------------------------------------------------')
print('Random Forest R2 degeri')
print(r2_score(Y, rf_reg.predict(X)))
print(r2_score(Y, rf_reg.predict(K)))
print(r2_score(Y, rf_reg.predict(Z)))



#Özet R2 değerleri
print('----------------------------------------------------------------')
print('Linear R2 degeri')
print(r2_score(Y, lin_reg.predict(X)))


print('----------------------------------------------------------------')
print('Polynomial R2 degeri')
print(r2_score(Y, lin_reg2.predict(poly_reg.fit_transform(X))))


print('----------------------------------------------------------------')
print('SVR R2 degeri')
print(r2_score(y_olcekli, svr_reg.predict(x_olcekli)))


print('----------------------------------------------------------------')
print('Decision Tree R2 degeri')
print(r2_score(Y, r_dt.predict(X)))


print('----------------------------------------------------------------')
print('Random Forest R2 degeri')
print(r2_score(Y, rf_reg.predict(X)))


print('----------------------------------------------------------------')