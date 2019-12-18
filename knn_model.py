__author__ = 'amir'

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

input_fields = ['launched_at_month', 'launched_at_year', 'category', 'parent_category', 'destination_delta_in_days', 'goal']
lower_bound = 1
upper_bound = 40

def run_model(df, k = None):
    X = df[input_fields]
    y = df['state']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
    scaler = StandardScaler()
    scaler.fit(X_train)

    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.3, random_state=1)

    error = []

    if k is None:
        # Calculating error for K values between 1 and 40
        for i in range(lower_bound, upper_bound+1):
            print('knn: ' + str(i))
            knn = KNeighborsClassifier(n_neighbors=i)
            knn.fit(X_train, y_train)
            pred_i = knn.predict(X_val)
            error.append(np.mean(pred_i != y_val))


        plt.figure(figsize=(12, 6))
        plt.plot(range(lower_bound, upper_bound+1), error, color='red', linestyle='dashed', marker='o',
                 markerfacecolor='blue', markersize=10)

        plt.title('Error Rate K Value')
        plt.xlabel('K Value')
        plt.ylabel('Mean Error Cross Validation')

        idx = error.index(min(error))
        k = idx + lower_bound
        print('chosen nearest neighbours: ' + str(k))
        plt.show()

    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    pred = knn.predict(X_test)
    print('precision is: ' + str(1-np.mean(pred != y_test)))
    return 1-np.mean(pred != y_test)



