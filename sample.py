# import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import prophet
print(prophet.__version__)

from sklearn.metrics import mean_absolute_error


# load dataset
path = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/monthly-car-sales.csv'
df = pd.read_csv(path, header=0)
df['Month'] = pd.to_datetime(df['Month'])
print(df.shape)
df.tail()


# buile Prophet model
df.columns = ['ds', 'y']  # Prophet model 需特定的欄位名稱 
train = df.drop(df.index[-12:])
model = prophet.Prophet()
model.fit(train)

# define in-sample testing set
future = list()
for i in range(1, 13):
    date = '1968-%02d' % i
    future.append([date])
    
future = pd.DataFrame(future, columns=['ds'])
future['ds']= pd.to_datetime(future['ds'])
future.head()

# do prediction
forecast = model.predict(future)
forecast.head()

# plot results
model.plot(forecast)
plt.scatter(x=df.ds, y=df.y)
plt.legend(['Actual', 'Predict'])
plt.show();