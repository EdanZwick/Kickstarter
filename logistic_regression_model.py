__author__ = 'amir'

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

input_fields = ['launched_at_month', 'launched_at_year', 'category', 'parent_category', 'destination_delta_in_months', 'goal']

def run_model(df):
    X = df[input_fields]
    y = df['state']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    print('running logistic regression no nlp')
    regression = LogisticRegression()
    regression.fit(X_train, y_train)
    pred = regression.predict(X_test)
    print('precision is: ' + str(1-np.mean(pred != y_test)))
    return (1-np.mean(pred != y_test))





