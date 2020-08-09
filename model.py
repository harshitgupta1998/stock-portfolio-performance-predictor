# -*- coding: utf-8 -*-
"""
Created on Sun Aug  9 12:02:25 2020

@author: harshit
"""

import pickle
import numpy as np
import pandas as pd
import numpy as np
import pandas as pd
import sklearn
import scipy
import matplotlib.pyplot as plt
import seaborn as sns
data=pd.read_csv(r'stock portfolio performance data set.csv')

data['Abs. Win Rate']=data['Abs. Win Rate'].str.strip('%').astype(float)
data['Total Risk']=data['Total Risk'].str.strip('%').astype(float)
data['Rel. Win Rate']=data['Rel. Win Rate'].str.strip('%').astype(float)
data['Annual Return']=data['Annual Return'].str.strip('%').astype(float)
data['Excess Return']=data['Excess Return'].str.strip('%').astype(float)

del data['Excess Return.1']
del data['Annual Return.1']
del data['Systematic Risk.1']
del data['Total Risk.1']
del data['Abs. Win Rate.1']
del data['Rel. Win Rate.1']
del data['ID']

from sklearn.model_selection import train_test_split
from pandas import DataFrame
#data.isnull().values.any()
#feature_columns=['Large B/P','Large ROE','Large S/P','Large Return Rate in the last quarter','Large Market Value']
X=data.iloc[:,0:5].values # check if the name is necessary
y1=data.iloc[:,5].values
y2=data.iloc[:,6].values
y3= data.iloc[:,7].values
y4=data.iloc[:,8].values
y5=data.iloc[:,9].values
y6=data.iloc[:,10].values
#print(x)
#print(DataFrame(y1)+DataFrame(y1))

#from sklearn.impute import SimpleImputer
#X_train, X_test, Y_train, Y_test= train_test_split(X,y1, test_size=0.30, random_state=10)
#fill_values = SimpleImputer(missing_values=0, strategy="mean")
#X_train = fill_values.fit_transform(X_train)
#X_test = fill_values.fit_transform(X_test)


from sklearn.linear_model import LinearRegression
regressor1 = LinearRegression()
regressor2 = LinearRegression()# try different models
regressor3 = LinearRegression()
regressor4 = LinearRegression()
regressor5 = LinearRegression()
regressor6 = LinearRegression()

#Fitting model with trainig data
regressor1.fit(X, y1)
regressor2.fit(X, y2)
regressor3.fit(X, y3)
regressor4.fit(X, y4)
regressor5.fit(X, y5)
regressor6.fit(X, y6)

# Saving model to disk
pickle.dump(regressor1, open('model1.pkl','wb'))
pickle.dump(regressor2, open('model2.pkl','wb'))
pickle.dump(regressor3, open('model3.pkl','wb'))
pickle.dump(regressor4, open('model4.pkl','wb'))
pickle.dump(regressor5, open('model5.pkl','wb'))
pickle.dump(regressor6, open('model6.pkl','wb'))


# Loading model to compare the results
model1 = pickle.load(open('model1.pkl','rb'))

model2 = pickle.load(open('model2.pkl','rb'))
model3 = pickle.load(open('model3.pkl','rb'))
model4 = pickle.load(open('model4.pkl','rb'))
model5 = pickle.load(open('model5.pkl','rb'))
model6 = pickle.load(open('model6.pkl','rb'))
print(model1.predict([[0.5,0, 0, 0,0.5]]))

print(model2.predict([[0.5,0, 0, 0,0.5]]))
print(model3.predict([[0.5,0, 0, 0,0.5]]))
print(model4.predict([[0.5,0, 0, 0,0.5]]))
print(model5.predict([[0.5,0, 0, 0,0.5]]))
print(model6.predict([[0.5,0, 0, 0,0.5]]))#0.5	0	0	0.5	0

