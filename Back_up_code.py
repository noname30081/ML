# import libraries
import datetime
from xmlrpc.client import DateTime
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import prophet
print(prophet.__version__)

from sklearn.metrics import mean_absolute_error


# load dataset
path = r'D:\Master Degree\Lesson\ML\HW1\program\ML\DataSet\training_data.csv'
#path = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/monthly-car-sales.csv'
df = pd.read_csv(path)
df = df.set_axis(['OPEN','MAX','MIN','CLOSE'],axis=1)

df.insert(0,'DATE',[datetime.date(2021,5,1)+datetime.timedelta(days=i) for i in range(0,len(df.index)) ])
df.insert(len(df.columns),'DEAL',[(df['CLOSE'][i] - df['OPEN'][i]) for i in range(0,len(df.index))])
df.insert(len(df.columns),'SUPPORT',[(df['CLOSE'][i] - 0.5*(df['MAX'][i]+df['MIN'][i])) for i in range(0,len(df.index))])
print(df)
print(df.shape)
df.tail()


# buile Prophet model
df2 = pd.DataFrame(df['CLOSE'].to_list(), columns=['y'])
#df2.insert(0,'ds',df['DATE'].to_list())
df2.insert(0,'ds',df['DATE'].to_list())

print(df2)


train = df2[0:30]
model = prophet.Prophet(changepoint_prior_scale= 0.9,interval_width=0.8,changepoint_range=1)
model.fit(train)

# define in-sample testing set
future = list()
for i in range(0, 32):
    date = datetime.date(2021,5,1)+datetime.timedelta(days=i)
    future.append([date])
    
future = pd.DataFrame(future, columns=['ds'])
future['ds']= pd.to_datetime(future['ds'])
future.head()

# do prediction
forecast = model.predict(future)
forecast.head()
print(forecast['yhat'][31])
print(df['SUPPORT'][30])
forecast['yhat'][30] = forecast['yhat'][30] + df['SUPPORT'][29]*-1
forecast['yhat'][31] = forecast['yhat'][31] + df['SUPPORT'][30]*-1

# plot results
model.plot(forecast)
plt.scatter(x=df2.ds, y=df2.y)
plt.legend(['Actual', 'Predict'])
plt.show();
input()