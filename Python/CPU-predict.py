import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from keras.models import *
from keras.layers import *
from keras.callbacks import *
from tensorflow import keras

def prepare_data(seq,num):
  x=[]
  y=[]
  for i in range(0,(len(seq)-num),1):
    
    input_ = seq[i:i+num]
    output  = seq[i+num]
    
    x.append(input_)
    y.append(output)
    
  return np.array(x), np.array(y)

data=pd.read_csv('Utilities\csv\metrics_2024-09-03\metrics.csv')
print(data.head)
cpu_usage = data['value'].values

num=20
#first 10 mins of traffic
sample = cpu_usage[:num]

x,y= prepare_data(cpu_usage,num)
print(len(x))

ind = int(0.9 * len(x))
x_tr = x[:ind]
y_tr = y[:ind]
x_val=x[ind:]
y_val=y[ind:]

#normalize the inputs
x_scaler= StandardScaler()
x_tr = x_scaler.fit_transform(x_tr)
x_val= x_scaler.transform(x_val)

#reshaping the output for normalization
y_tr=y_tr.reshape(len(y_tr),1)
y_val=y_val.reshape(len(y_val),1)

#normalize the output
y_scaler=StandardScaler()
y_tr = y_scaler.fit_transform(y_tr)[:,0]
y_val = y_scaler.transform(y_val)[:,0]

#reshaping input data
x_tr= x_tr.reshape(x_tr.shape[0],x_tr.shape[1],1)
x_val= x_val.reshape(x_val.shape[0],x_val.shape[1],1)
print(x_tr.shape)

# define model
model =  Sequential()
model.add(LSTM(128,input_shape=(104,1)))
model.add(Dense(64,activation='relu'))
model.add(Dense(1,activation='linear'))
model.summary()