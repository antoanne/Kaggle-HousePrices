#!/usr/bin/python
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics

import math


def corr(data1, data2):

    mean1 = data1.mean()
    mean2 = data2.mean()
    std1 = data1.std()
    std2 = data2.std()

    corr = ((data1*data2).mean()-mean1*mean2)/(std1*std2)
    return corr

def average(x):
    assert len(x) > 0
    return float(sum(x)) / len(x)

def pearson_def(x, y):
    assert len(x) == len(y)
    n = len(x)
    assert n > 0
    avg_x = average(x)
    avg_y = average(y)
    diffprod = 0
    xdiff2 = 0
    ydiff2 = 0
    for idx in range(n):
        xdiff = x[idx] - avg_x
        ydiff = y[idx] - avg_y
        diffprod += xdiff * ydiff
        xdiff2 += xdiff * xdiff
        ydiff2 += ydiff * ydiff

    return diffprod / math.sqrt(xdiff2 * ydiff2)

variavel_independente = 'OverallQual'

# 'Fireplaces', '3SsnPorch','ScreenPorch'

#[['LotArea','YearBuilt', '1stFlrSF', '2ndFlrSF','GrLivArea','HalfBath','FullBath',
#'TotRmsAbvGrd','YearRemodAdd','BedroomAbvGr','OverallQual','OverallCond','GarageArea',
# 'BsmtFinSF1','BsmtUnfSF','TotalBsmtSF','GarageCars','WoodDeckSF','OpenPorchSF']]

dataHouse= pd.read_csv("desafio_house_train.csv")
'''
dataHouse = dataHouse.loc[dataHouse['LotArea'] < 20000]
dataHouse = dataHouse.loc[dataHouse['SalePrice'] < 350000]
dataHouse = dataHouse.loc[dataHouse['SalePrice'] > 48000]
'''

X = dataHouse[[variavel_independente]]
y = dataHouse['SalePrice']


#separando a base em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

#fazendo a regressão
regressor = LinearRegression()
regressor.fit(X_train, y_train)

#apresentando coeficientes
coeff_df = pd.DataFrame(regressor.coef_, X.columns, columns=['Coefficient'])
print('\n')
print('Coeficientes:')
print(coeff_df)


#fazendo predições
y_pred = regressor.predict(X_test)
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
df  = df.sort_values(by=['Actual'])
print('\n')
print('Previsões:')
print (df)

#calculando métricas
print('\n')
print('Métricas:')
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % metrics.r2_score(y_test, y_pred))


#Preparando os dados para plotagem -------------------------
X = dataHouse[variavel_independente]
y = dataHouse['SalePrice']

#Calculando o Coeficiente de Correlação de Pearsom por dois algoritmos
X=X.values.reshape(len(X),1)
y=y.values.reshape(len(y),1)
print ('Pearson')
print(pearson_def(X, y))
print(corr(X, y))

