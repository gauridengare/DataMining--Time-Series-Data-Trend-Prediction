import pandas as pd
from statsmodels.tsa.arima_model import ARIMA
import numpy as np
import math

#opening a file and writing predictions to that file
file = open('output.txt','w+')

#Overall predictions for 100 products
series = pd.read_csv('training.csv', header=None,index_col=0)
array = series.to_xarray()
#data list has overall sale for each product
data=[]
 #printing key id as 0 for overall prediction
file.write('0 ')
#adding column values to get overall sale for each product
for i in array.values():
    X = i.values
    sum=np.sum(X)
    data.append(sum)
#predictions for next 29 days
for t in range(29):
        model = ARIMA(data, order=(7,1,1))
        model_fit = model.fit(disp=False)
        out = model_fit.forecast()
        pred = out[0]
        data.append(pred)
        file.write('%.f  ' % (pred))
file.write('\n')


#prediction for each product for each day
series=pd.read_csv('training.csv', header=None,index_col=None)
forid=series.transpose()
SeriesWithoutIds=series.drop(labels=None, axis=0, index=None, columns=0, level=None, inplace=False, errors='raise')
TransSeries=SeriesWithoutIds.transpose()
array = TransSeries.to_xarray()
KeyIds = forid.as_matrix()
c = 0
for i in array.values():
    X = i.values
    data = X.tolist()
    #printing key ids
    print('%s  ' % KeyIds[0][c], end='')
    file.write('%s ' % KeyIds[0][c])
    #predictions for next 29 days for each product
    for t in range(29):
        model = ARIMA(data, order=(6,1,0))
        model_fit = model.fit(disp=False)
        out = model_fit.forecast()
        pred = out[0]
        #if predicted value is negative, making it 0
        if pred<0 :
            pred=0
        data.append(pred)
        #rounding off the predicted value to nearest integer
        pred=math.floor(pred)
        file.write('%.f  ' % (pred))
    c += 1
    file.write('\n')
file.close()
