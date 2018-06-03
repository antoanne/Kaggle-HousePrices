#!/usr/bin/python
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics


dataHouse= pd.read_csv("desafio_house_train.csv")
dataHouse = dataHouse.loc[dataHouse['LotArea'] < 20000]
dataHouse = dataHouse.loc[dataHouse['SalePrice'] < 350000]
dataHouse = dataHouse.loc[dataHouse['SalePrice'] > 48000]

#X=dataHouse[['LotArea']]
#X=dataHouse[['LotArea', 'OverallQual']]
#X=dataHouse[['LotArea', 'OverallQual','GrLivArea']]
#X=dataHouse[['LotArea', 'OverallQual','GrLivArea','GarageCars']]
#X=dataHouse[['LotArea', 'OverallQual','GrLivArea','GarageCars','GarageArea']]
#X=dataHouse[['LotArea', 'OverallQual','GrLivArea','GarageCars','GarageArea','FullBath']]
#X=dataHouse[['LotArea', 'OverallQual','GrLivArea','GarageCars','GarageArea','FullBath','TotalBsmtSF']]
#X=dataHouse[['LotArea', 'OverallQual','GrLivArea','GarageCars','GarageArea','FullBath','TotalBsmtSF', 'YearBuilt']]
#X=dataHouse[['LotArea', 'OverallQual','GrLivArea','GarageCars','GarageArea','FullBath','TotalBsmtSF', 'YearBuilt','1stFlrSF']]
#X=dataHouse[['LotArea', 'OverallQual','GrLivArea','GarageCars','GarageArea','FullBath','TotalBsmtSF', 'YearBuilt','1stFlrSF','YearRemodAdd']]
#X=dataHouse[['LotArea', 'OverallQual','GrLivArea','GarageCars','GarageArea','FullBath','TotalBsmtSF', 'YearBuilt','1stFlrSF','YearRemodAdd', 'Fireplaces']]
#X=dataHouse[['LotArea', 'OverallQual','GrLivArea','GarageCars','GarageArea','FullBath','TotalBsmtSF', 'YearBuilt','1stFlrSF','YearRemodAdd','Fireplaces','OpenPorchSF']]
#X=dataHouse[['LotArea', 'OverallQual','GrLivArea','GarageCars','GarageArea','FullBath','TotalBsmtSF', 'YearBuilt','1stFlrSF','YearRemodAdd','Fireplaces','OpenPorchSF','2ndFlrSF']]
#X=dataHouse[['LotArea', 'OverallQual','GrLivArea','GarageCars','GarageArea','FullBath','TotalBsmtSF', 'YearBuilt','1stFlrSF','YearRemodAdd','Fireplaces','OpenPorchSF','2ndFlrSF','BsmtFinSF1']]
#X=dataHouse[['LotArea', 'OverallQual','GrLivArea','GarageCars','GarageArea','FullBath','TotalBsmtSF', 'YearBuilt','1stFlrSF','YearRemodAdd','Fireplaces','OpenPorchSF','2ndFlrSF','BsmtFinSF1','HalfBath']]
#X=dataHouse[['LotArea', 'OverallQual','GrLivArea','GarageCars','GarageArea','FullBath','TotalBsmtSF', 'YearBuilt','1stFlrSF','YearRemodAdd','Fireplaces','OpenPorchSF','2ndFlrSF','BsmtFinSF1','HalfBath','BsmtUnfSF']]
X=dataHouse[['LotArea', 'OverallQual','GrLivArea','GarageCars','GarageArea','FullBath','TotalBsmtSF', 'YearBuilt','1stFlrSF','YearRemodAdd','Fireplaces','OpenPorchSF','2ndFlrSF','BsmtFinSF1','HalfBath','BsmtUnfSF','BedroomAbvGr']]

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
