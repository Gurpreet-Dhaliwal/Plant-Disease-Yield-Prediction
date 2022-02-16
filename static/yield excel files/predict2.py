import pandas as pd
name = 'abc.csv'
state= 'WEST BENGAL'
print("For state west bengal")
crops= ['Yield_rice', 'Yield_cotton', 'Yield_oilseeds', 'Yield_pulses']
cropindex_dict= {'Yield_rice':2, 'Yield_wheat':3, 'Yield_sugarcane':4, 'Yield_cotton':5, 'Yield_oilseeds':6, 'Yield_pulses':7}
price=[1700,5150,3300,6000]
attributes= ['rain_Jan', 'rain_Feb', 'rain_Mar', 'rain_Apr', 'rain_May', 'rain_Jun', 'rain_Jul', 'rain_Aug', 'rain_Sep', 'rain_Oct', 'rain_Nov', 'rain_Dec', 'cc_Jan', 'cc_Feb', 'cc_Mar', 'cc_Apr', 'cc_May', 'cc_Jun', 'cc_Jul', 'cc_Aug', 'cc_Sep', 'cc_Oct', 'cc_Nov', 'cc_Dec', 'max_temp_Jan', 'max_temp_Feb', 'max_temp_Mar', 'max_temp_Apr', 'max_temp_May', 'max_temp_Jun', 'max_temp_Jul', 'max_temp_Aug', 'max_temp_Sep', 'max_temp_Oct', 'max_temp_Nov', 'max_temp_Dec', 'min_temp_Jan', 'min_temp_Feb', 'min_temp_Mar', 'min_temp_Apr', 'min_temp_May', 'min_temp_Jun', 'min_temp_Jul', 'min_temp_Aug', 'min_temp_Sep', 'min_temp_Oct', 'min_temp_Nov', 'min_temp_Dec']
states=['HIMACHAL PRADESH', 'BIHAR', 'KERALA', 'WEST BENGAL', 'GUJARAT', 'PUNJAB', 'UTTAR PRADESH', 'TAMIL NADU']
ind= [2,5,6,7]
kharif_crops= ['Yield_rice',  'Yield_cotton', 'Yield_oilseeds', 'Yield_pulses']
rabi_crops=[ 'Yield_wheat' ]
yearly_crops= [ 'Yield_sugarcane' ]
# Importing the dataset
list=[]
dataset = pd.read_csv(name)
df= pd.DataFrame(dataset)
dataset= df.loc[df['State']==state]
y = dataset.iloc[:, ind].values
dict_price={}
for i in range(0,len(y[0])):
    dict_price[crops[i]]=(price[i]*y[0][i])/10
dict_yield = {}
for i in range(0,len(y[0])):
    dict_yield[crops[i]]=y[0][i]

import operator
sorted_x = sorted(dict_yield.items(), key=operator.itemgetter(0))
print(sorted_x)
dict_yield_final= {}

print(dict_yield_final)
for t in sorted_x:
    print(t[0])


"""
datafr= pd.DataFrame(list)
datafr.to_csv("abc.csv", sep=',')
"""