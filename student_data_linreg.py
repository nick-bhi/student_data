# -*- coding: utf-8 -*-
"""
Created on Fri Jan  3 02:21:43 2020

@author: Nick
"""

import numpy as np
import pandas as pd
import sklearn
from sklearn import linear_model

df = pd.read_csv("student-mat.csv", sep = ';')
df.head()

data = df[['G2','schoolsup','school','activities','internet','famrel']]
output = 'G3'

columns = list(data)
for i in columns:
    if data[i].dtype == 'O':
        data[i] = data[i].astype('category')
        data[i] = data[i].cat.codes
        
list_data = list(data)

X = np.array(data)
Y = np.array(df[output])

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X,Y, test_size = 0.1)
linear = linear_model.LinearRegression()
linear.fit(x_train, y_train)
R2 = linear.score(x_test, y_test)

coeff_dict = {}
print("R squared:", R2)
print('Intercept:', linear.intercept_)
print('Coefficients: \n')
for i in range(len(linear.coef_)):
    coeff_dict[list_data[i]] = linear.coef_[i]
for i in range(len(coeff_dict)):
    value = sorted(coeff_dict.values())[i]
    key = ''
    for j in coeff_dict.keys():
        if coeff_dict[j] == value:
            key = j
    print(key,":",value)