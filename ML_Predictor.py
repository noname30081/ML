# import libraries
import datetime
from xmlrpc.client import DateTime
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import prophet
print(prophet.__version__) 

from sklearn.metrics import mean_absolute_error

modelStartDay = datetime.date(2021,5,1)
basic_status = 0
TrainDatas = pd.DataFrame()
model = prophet.Prophet(changepoint_prior_scale= 0.9,interval_width=0.8,changepoint_range=1)

High_n=0.0#最近最高價
Low_n=0.0#最近最低價

def train(train_data) :
    global modelStartDay
    global TrainDatas
    global model

    train_data.insert(0,'DATE',[modelStartDay+datetime.timedelta(days=i) for i in range(0,len(train_data.index)) ])
    train_data.insert(len(train_data.columns),'DEAL',[(train_data['CLOSE'][i] - train_data['OPEN'][i]) for i in range(0,len(train_data.index))])
    train_data.insert(len(train_data.columns),'SUPPORT',[(train_data['CLOSE'][i] - 0.5*(train_data['MAX'][i]+train_data['MIN'][i])) for i in range(0,len(train_data.index))])
    train_data.tail()
    TrainDatas = train_data
    print(f"{train_data}")

    #using OPEN PRICE for Training\Prediction
    df2 = pd.DataFrame(train_data['OPEN'].to_list(), columns=['y'])
    df2.insert(0,'ds',train_data['DATE'].to_list())
    model.fit(df2)

def predict_action(row_data) :
    global modelStartDay
    global TrainDatas
    global model
    global basic_status

    future = list()
    date = modelStartDay+datetime.timedelta(days=len(TrainDatas.index))
    future.append([date])
        
    future = pd.DataFrame(future, columns=['ds'])
    future['ds']= pd.to_datetime(future['ds'])
    future.head()

    # do prediction
    forecast = model.predict(future)
    forecast.head()
    forecast['yhat'][0] = forecast['yhat'][0] + TrainDatas['SUPPORT'][len(TrainDatas.index)-1]*1.5

    act = 0
    if(forecast['yhat'][0] <= TrainDatas['CLOSE'][len(TrainDatas.index)-1]):
        if(basic_status == -1) :
            basic_status = -1
            act = 0
        elif(basic_status == 0) :
            basic_status = -1
            act = -1
        else :
            basic_status = 0
            act = -1
    elif(forecast['yhat'][0] >= TrainDatas['CLOSE'][len(TrainDatas.index)-1]) :
        if(basic_status == 0) :
            basic_status = 1
            act = 1
        elif(basic_status == -1) :
            basic_status = 0
            act = 1
        else :
            basic_status = 1
            act = 0
    else :
        act = 0
    a = forecast['yhat'][0]
    b = TrainDatas['CLOSE'][len(TrainDatas.index)-1]
    print(f"{a} : {b} [{act}]")


    pdtemp = pd.DataFrame()
    pdtemp = pdtemp.append(pd.Series(row_data[1]),ignore_index=True)
    pdtemp.insert(0,'DATE',[date])
    pdtemp.insert(len(pdtemp.columns),'DEAL',[(pdtemp['CLOSE'][i] - pdtemp['OPEN'][i]) for i in range(0,len(pdtemp.index))])
    pdtemp.insert(len(pdtemp.columns),'SUPPORT',[(pdtemp['CLOSE'][i] - 0.5*(pdtemp['MAX'][i]+pdtemp['MIN'][i])) for i in range(0,len(pdtemp.index))])
    pdtemp.tail()

    TrainDatas = TrainDatas.append(pdtemp,ignore_index=True)
    print(f"LEN OF TRAIN : {len(TrainDatas.index)}")
    return act

def re_training() :
    global TrainDatas
    global model

    model = prophet.Prophet(changepoint_prior_scale= 0.9,interval_width=0.8,changepoint_range=1)
    #using OPEN PRICE for Training\Prediction
    df2 = pd.DataFrame(TrainDatas['OPEN'].to_list(), columns=['y'])
    df2.insert(0,'ds',TrainDatas['DATE'].to_list())
    model.fit(df2)

'''
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
df2 = pd.DataFrame(df['OPEN'].to_list(), columns=['y'])
#df2.insert(0,'ds',df['DATE'].to_list())
df2.insert(0,'ds',df['DATE'].to_list())

print(df2)

forecast = pd.DataFrame()
for dateTras in range(0,int(len(df.index)*0.3)) :
    train = df2[0:30+dateTras]
    model = prophet.Prophet(changepoint_prior_scale= 0.9,interval_width=0.8,changepoint_range=1)
    model.fit(train)

    # define in-sample testing set
    future = list()
    date = datetime.date(2021,5,1)+datetime.timedelta(days=dateTras)
    future.append([date])
        
    future = pd.DataFrame(future, columns=['ds'])
    future['ds']= pd.to_datetime(future['ds'])
    future.head()

    # do prediction
    if(dateTras == 0) :
        forecast = model.predict(future)
        forecast.head()
    else :
        forecast_temp = model.predict(future)
        forecast_temp.head()
        forecast = forecast.append(forecast_temp,ignore_index=True)
    forecast['yhat'][dateTras] = forecast['yhat'][dateTras] + df['SUPPORT'][30+dateTras]*1.5
    print(dateTras)

    # plot results
model.plot(forecast)
plt.scatter(x=df2.ds, y=df2.y)
plt.legend(['Actual', 'Predict'])
plt.show();

input()
'''