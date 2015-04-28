import profile
import numpy as np
import pandas as pd
from sklearn import preprocessing
from OnlineClassificationTree import *

def test():
    filename = "dataset.csv"
    df = pd.read_csv(filename, header = 0)
    data = df.values
    y = data[:, -1]
    lbl_enc = preprocessing.LabelEncoder()
    y = lbl_enc.fit_transform(y)
    data = data[:, 0:-1]
    train = data[0:50000]
    ytrain = y[0:50000]
    test = data[50000:]
    ytest = y[50000:]
    learner = ClassificationTree(number_of_features=93)
    
    for t, x in enumerate(train):
        learner.update(x, ytrain[t])
        if t % 1000 == 0:
            print t
    correct_num = 0
    for t, x in enumerate(test):
        y_pred = learner.predict(x)
        if y_pred == ytest[t]:
            correct_num += 1
        if t % 1000 == 0:
            print t

    print correct_num

if __name__ == '__main__':
    profile.run("test()")
