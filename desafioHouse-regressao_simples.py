#!/usr/bin/python
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics

variavel_independente = 'LotArea'

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

#separando a base em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

X=X.values.reshape(len(X),1)
y=y.values.reshape(len(y),1)
X_train=X_train.values.reshape(len(X_train),1)
y_train=y_train.values.reshape(len(y_train),1)
X_test=X_test.values.reshape(len(X_test),1)
y_test=y_test.values.reshape(len(y_test),1)

### Construindo gráficos (válido para a variável lot Área
### Para as demais variáveis deve-se substituri o xticks, yticks por (())


### Plotando todos os dados juntos
plt.scatter(X, y,  color='dodgerblue', s=2)
#plt.scatter(X_test, y_test,  color='lightgreen', s=2)
#linha de regressão
plt.plot(X_test, y_pred, color='darkorange',linewidth=2)
#plt.scatter(X_test, y_pred,  color='darkorange', s=2)
plt.title('Linha de Regressão - Base Completa')
plt.xlabel('Área do Imóvel')
plt.ylabel('Preço')
xint = [10000,50000,100000,150000,200000,250000]
yint = [100000,200000,300000,400000,500000,600000,700000,800000]
plt.xticks(xint)
plt.yticks(yint)
plt.show()

plt.hist(y, color='dodgerblue', bins=50)
plt.xticks(yint)
plt.yticks(())
plt.title('Distribuição da variável dependente (preço) - Base Completa')
plt.show()

plt.hist(X,color='dodgerblue', bins=50)
plt.xticks(xint)
plt.yticks(())
plt.title('Distribuição da variável independente (área) - Base Completa')
plt.show()

