__author__ = 'amir'

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

input_fields = ['launched_at_month', 'launched_at_year', 'category', 'parent_category', 'destination_delta_in_days', 'goal', 'nima_score']
lower_bound = 100
upper_bound = 1000
step = 50

def run_model(df, num_trees = None, nima = False, input_fields = None):
    if input_fields == None:
        input_fields = ['launched_at_month', 'launched_at_year', 'category', 'parent_category', 'destination_delta_in_days', 'goal']
    if nima:
        input_fields += ['nima_score', 'nima_tech']
    cols = list(df.columns.values)
    fields = [field for field in input_fields if field in cols]
    X = df[fields]
    y = df['state']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
    scaler = StandardScaler()
    scaler.fit(X_train)

    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.3, random_state=1)

    if num_trees is None:
        error = []
        # Calculating error for number of trees between 1 and 100
        for i in range(lower_bound, upper_bound+1, step):
            print('forest: ' + str(i))
            forest = RandomForestClassifier(n_estimators=i)
            forest.fit(X_train, y_train)
            pred_i = forest.predict(X_val)
            error.append(np.mean(pred_i != y_val))


        plt.figure(figsize=(12, 6))
        plt.plot(range(lower_bound, upper_bound+1, step), error, color='red', linestyle='dashed', marker='o',
                 markerfacecolor='blue', markersize=10)

        plt.title('Error Rate Number of trees')
        plt.xlabel('Number of trees')
        plt.ylabel('Mean Error Cross Validation')

        idx = error.index(min(error))

        print('chosen number of trees: ' + str(lower_bound + idx *step))
        num_trees = lower_bound + idx *step

    forest = RandomForestClassifier(n_estimators=num_trees)
    forest.fit(X_train, y_train)
    pred = forest.predict(X_test)
    print('precision is: ' + str(1-np.mean(pred != y_test)))
    plt.show()
    return (1-np.mean(pred != y_test))




