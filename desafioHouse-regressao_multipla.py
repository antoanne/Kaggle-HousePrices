#!/usr/bin/python
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics


dataHouse= pd.read_csv("desafio_house_train.csv")

#X = dataHouseTrain[['LotArea']]
#X = dataHouseTrain[['LotArea', 'YearBuilt']]
#X = dataHouseTrain[['LotArea', 'YearBuilt', 'YearRemodAdd']]
X = dataHouse[['LotArea', 'YearBuilt', 'YearRemodAdd', 'BedroomAbvGr']]

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
