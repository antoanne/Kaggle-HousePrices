#!/usr/bin/python
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
import pandas as pd


dataHouseTrain = pd.read_csv("dataHouse/train.csv")
dataHouseTest = pd.read_csv("dataHouse/test.csv")

### LotArea x SalePrice
### Carregando dados da base de treino
X = dataHouseTrain['LotArea']
Y = dataHouseTrain['SalePrice']
### Normalizando
X = X.values.reshape(len(X),1)
Y = Y.values.reshape(len(Y),1)
### Inicializando modelo com dados de treino
regr = linear_model.LinearRegression()
regr.fit(X, Y)
### Carregando dados para teste
X_test = dataHouseTest['LotArea']
### Normalizando
X_test=X_test.values.reshape(len(X_test),1)
### Gerando previsão de preços com base na regressão linear
#Y_predicted = regr.predict(X_test)
dataHouseTest['SalePrice'] = regr.predict(X_test)
### Construindo gráficos
### Plotando dados de treino
plt.scatter(X, Y,  color='blue', s=2)
### Plotando dados gerados com previsão
plt.scatter(dataHouseTest['LotArea'], dataHouseTest['SalePrice'],  color='black', s=2)
plt.title('House Data')
plt.xlabel('Size')
plt.ylabel('Price')
#plt.xticks(())
#plt.yticks(())
### Plotando regressão linear
plt.plot(dataHouseTest['LotArea'], dataHouseTest['SalePrice'], color='red',linewidth=2)
plt.show()

### Train
dataHouseTrain['SalePrice'].describe()
### Test
dataHouseTest['SalePrice'].describe()