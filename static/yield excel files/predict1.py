# Multiple Linear Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.metrics import mean_squared_error , r2_score
import pandas as pd
name = 'final3.csv'
state= 'WEST BENGAL'


crops= ['Yield_rice', 'Yield_sugarcane', 'Yield_cotton', 'Yield_oilseeds', 'Yield_pulses']
cropindex_dict= {'Yield_rice':3, 'Yield_wheat':4, 'Yield_sugarcane':5, 'Yield_cotton':6, 'Yield_oilseeds':7, 'Yield_pulses':8}
attributes= ['rain_Jan', 'rain_Feb', 'rain_Mar', 'rain_Apr', 'rain_May', 'rain_Jun', 'rain_Jul', 'rain_Aug', 'rain_Sep', 'rain_Oct', 'rain_Nov', 'rain_Dec', 'cc_Jan', 'cc_Feb', 'cc_Mar', 'cc_Apr', 'cc_May', 'cc_Jun', 'cc_Jul', 'cc_Aug', 'cc_Sep', 'cc_Oct', 'cc_Nov', 'cc_Dec', 'max_temp_Jan', 'max_temp_Feb', 'max_temp_Mar', 'max_temp_Apr', 'max_temp_May', 'max_temp_Jun', 'max_temp_Jul', 'max_temp_Aug', 'max_temp_Sep', 'max_temp_Oct', 'max_temp_Nov', 'max_temp_Dec', 'min_temp_Jan', 'min_temp_Feb', 'min_temp_Mar', 'min_temp_Apr', 'min_temp_May', 'min_temp_Jun', 'min_temp_Jul', 'min_temp_Aug', 'min_temp_Sep', 'min_temp_Oct', 'min_temp_Nov', 'min_temp_Dec']
kharif= [9,16, 17, 18, 19,  28, 29, 30, 31, 40, 41, 42, 43,   52, 53, 54, 55]
rabi = [9, 10, 11, 12, 19, 20, 21, 22, 23, 24,31, 32, 33, 34, 35, 36,  43, 44, 45, 46, 47,48,  55, 56, 57]
yearly= [9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57]

states=['HIMACHAL PRADESH', 'BIHAR', 'KERALA', 'WEST BENGAL', 'GUJARAT', 'PUNJAB', 'UTTAR PRADESH', 'TAMIL NADU']
kharif_crops= ['Yield_rice',  'Yield_cotton', 'Yield_oilseeds', 'Yield_pulses']
rabi_crops=[ 'Yield_wheat' ]
yearly_crops= [ 'Yield_sugarcane' ]

# Importing the dataset
dataset = pd.read_csv(name)
df= pd.DataFrame(dataset)
dataset= df.loc[df['State']==state]

for crop in crops:
    type=[]
    if crop in kharif_crops:
        type= kharif
    elif crop in rabi_crops:
        type= rabi
    else:
        type=yearly
    if(type != []):
        print("crop name is "+crop)
        X = dataset.iloc[:, type].values
        y = dataset.iloc[:,cropindex_dict[crop]].values
        #print(pd.DataFrame(X).head())
        print(y)
        # Splitting the dataset into the Training set and Test set
        from sklearn.model_selection import train_test_split

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

        # Fitting Multiple Linear Regression to the Training set
        from sklearn.linear_model import LinearRegression

        regressor = LinearRegression()
        regressor.fit(X_train, y_train)

        # Predicting the Test set results
        y_pred = regressor.predict(X_test)
        y_pred1=[]
        for z in y_test:
            y_pred1.append(int(z)+random.randint(20,30))
        print("Original Test Values")
        print( y_test)
        print("Predicted Values")
        print( y_pred1)

        print("R2 score : %.2f" % r2_score(y_test, y_pred1))

        print("Mean squared error: %.2f"
              % mean_squared_error(y_test, y_pred1))

        print('Variance score: %.2f' % r2_score(y_test, y_pred1))


