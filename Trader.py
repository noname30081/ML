import pandas as pd
import csv

def load_data(source) :
    df = pd.read_csv(source)
    cols = df.columns.to_list()
    dftemp = pd.Series({'OPEN':float(cols[0]),'MAX':float(cols[1]),'MIN':float(cols[2]),'CLOSE':float(cols[3])})
    df1 = pd.DataFrame()
    df1 = df1.append(dftemp,ignore_index=True)
    df= df.set_axis(['OPEN','MAX','MIN','CLOSE'],axis=1)
    df1 = df1.append(df,ignore_index=True)
    print(df1)
    return df1

if __name__ == '__main__':
     # You should not modify this part.
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--training',
    default='training_data.csv',
    help='input training data file name')
    parser.add_argument('--testing',
    default='testing_data.csv',
    help='input testing data file name')
    parser.add_argument('--output',
    default='output.csv',
    help='output file name')
    args = parser.parse_args()

    import ML_Predictor as trader
    # The following part is an example.
    # You can modify it at will.
    #training_data = load_data(args.training)
    training_data = load_data(r'D:\Master Degree\Lesson\ML\HW1\program\ML\DataSet\training_data.csv')
    trader.train(training_data)

    #testing_data = load_data(args.testing)
    testing_data = load_data(r'D:\Master Degree\Lesson\ML\HW1\program\ML\DataSet\testing_data.csv')
    with open(args.output, 'w') as output_file:
        for row in testing_data.iterrows():
            #print(testing_data)
            #print(type(row))
            # We will perform your action as the open price in the next day.
            action = trader.predict_action(row)
            output_file.write(str(action)+'\n')

            # this is your option, you can leave it empty.
            trader.re_training()

