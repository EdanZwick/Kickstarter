__author__ = 'amir'

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier

input_fields = ['launched_at_month', 'launched_at_year', 'category', 'parent_category', 'destination_delta_in_days', 'goal']
lower_bound = 100
upper_bound = 1000
step = 50

def run_model(df, estimators = None):

    X = df[input_fields]
    y = df['state']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
    scaler = StandardScaler()
    scaler.fit(X_train)

    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.3, random_state=1)

    if estimators is None:
        error = []
        # Calculating error for number of trees between 1 and 100
        for i in range(lower_bound, upper_bound+1, step):
            print('boosting: ' + str(i))
            forest = GradientBoostingClassifier(n_estimators=i)
            forest.fit(X_train, y_train)
            pred_i = forest.predict(X_val)
            error.append(np.mean(pred_i != y_val))


        plt.figure(figsize=(12, 6))
        plt.plot(range(lower_bound, upper_bound+1, step), error, color='red', linestyle='dashed', marker='o',
                 markerfacecolor='blue', markersize=10)

        plt.title('Error Rate Number of estimators')
        plt.xlabel('Number of estimators')
        plt.ylabel('Mean Error Cross Validation')

        idx = error.index(min(error))

        print('chosen number of estimators: ' + str(lower_bound + idx *step))
        estimators = lower_bound + idx *step

    forest = GradientBoostingClassifier(n_estimators=estimators)
    forest.fit(X_train, y_train)
    pred = forest.predict(X_test)
    print('precision is: ' + str(1-np.mean(pred != y_test)))
    plt.show()
    return (1 - np.mean(pred != y_test))




