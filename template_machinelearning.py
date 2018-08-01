# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

'''
Importing Data
'''
data_sales = pd.read_csv('sales.csv')
#TODO convrert yes/no to boolean
X_sales_raw = data_sales.iloc[:, :-1].values
y_sales_raw = data_sales.iloc[:,3].values

'''
Imputer
'''
from sklearn.preprocessing import Imputer
imputer_sales = Imputer(missing_values='NaN',strategy='mean',axis =0)
imputer_sales.fit(X_sales[:,1:3])
X_sales[:,1:3] = imputer_sales.transform(X_sales[:,1:3])

'''
Encoding 
'''
#labelencoding + OneHotEncoder
#for the countries we have to first LabelEncode it and then do a OneHotEncoder
from sklearn.preprocessing import LabelEncoder
labelenconder_sales = LabelEncoder()
X_sales[:,0] = labelenconder_sales.fit_transform(X_sales_raw[:,0])
from sklearn.preprocessing import OneHotEncoder
onehotencoder = OneHotEncoder(categorical_features=[0])
X_sales = onehotencoder.fit_transform(X_sales).toarray()

#for the yes, no we only have to LabelEncode it
labelenconder_sales_y = LabelEncoder()
y_sales = labelenconder_sales_y.fit_transform(y_sales_raw)

'''
Splitting data
'''
# depricated: from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_sales, y_sales_raw, test_size = 0.2, random_state = 0)

'''
Feature Scaler
'''
#should we scale dummy variable
from sklearn.preprocessing import StandardScaler
sc_X =StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

