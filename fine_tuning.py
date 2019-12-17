import numpy as np
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

from data_loader import get_data

N_SPLITS = 5


def _scale(*args):
    scaler = StandardScaler()
    scaler.fit(args[0])
    return map(scaler.transform, args)


def _get_kfold(data, index):
    return data.train_x.iloc[index], data.train_y[index]


def fine_tune_knn():
    data = get_data()
    base = 1
    n = 30
    for i, k in enumerate(range(base, base + n, 2)):
        for scale in [False]:  # NOTE: Scaling is bad for knn!
            kf = KFold(n_splits=N_SPLITS)
            acc = 0
            for train_index, val_index in kf.split(data.train_x):
                knn = KNeighborsClassifier(n_neighbors=k)
                train_x, train_y = _get_kfold(data, train_index)
                val_x, val_y = _get_kfold(data, val_index)
                if scale:
                    train_x, val_x = _scale(train_x, val_x)
                knn.fit(train_x, train_y)
                acc += accuracy_score(val_y, knn.predict(val_x))
            acc /= N_SPLITS
            print(f"knn: Accuracy for k={k}{' with scaler ' if scale else ''} is {acc * 100}%")


def vanila_forest():
    data = get_data()
    for scale in [True, False]:
        for estimators in range(200, 600, 150):
            for criterion in ["gini", "entropy"]:
                kf = KFold()
                acc = 0
                for train_index, val_index in kf.split(data.train_x):
                    forest = RandomForestClassifier(n_estimators=estimators, criterion=criterion, random_state=42)
                    train_x, train_y = _get_kfold(data, train_index)
                    val_x, val_y = _get_kfold(data, val_index)
                    if scale:
                        train_x, val_x = _scale(train_x, val_x)
                    forest.fit(train_x, train_y)
                    acc += accuracy_score(val_y, forest.predict(val_x))
                acc /= N_SPLITS
                print(
                    f"RandomForest: Accuracy for params={(estimators, 'no_depth', criterion)}{' with scaler ' if scale else ''} is {acc * 100}%")


def fine_tune_forest():
    data = get_data()
    for scale in [True, False]:
        for estimators in range(200, 600, 150):
            for depth in range(3, 7):
                for criterion in ["gini", "entropy"]:
                    kf = KFold()
                    acc = 0
                    for train_index, val_index in kf.split(data.train_x):
                        forest = RandomForestClassifier(n_estimators=estimators, criterion=criterion, max_depth=depth,
                                                        random_state=42)
                        train_x, train_y = _get_kfold(data, train_index)
                        val_x, val_y = _get_kfold(data, val_index)
                        if scale:
                            train_x, val_x = _scale(train_x, val_x)
                        forest.fit(train_x, train_y)
                        acc += accuracy_score(val_y, forest.predict(val_x))
                    acc /= N_SPLITS
                    print(
                        f"RandomForest: Accuracy for params={(estimators, depth, criterion)}{' with scaler ' if scale else ''} is {acc * 100}%")


def fine_tune_gb():
    data = get_data()
    for scale in [True, False]:
        for learn_rate in np.arange(0.02, 0.15, 0.04):
            for depth in range(2, 6):
                kf = KFold()
                acc = 0
                for train_index, val_index in kf.split(data.train_x):
                    gbc = GradientBoostingClassifier(learning_rate=learn_rate, max_depth=depth, random_state=42)
                    train_x, train_y = _get_kfold(data, train_index)
                    val_x, val_y = _get_kfold(data, val_index)
                    if scale:
                        train_x, val_x = _scale(train_x, val_x)
                    gbc.fit(train_x, train_y)
                    acc += accuracy_score(val_y, gbc.predict(val_x))
                acc /= N_SPLITS
                print(
                    f"GradientBoosting: Accuracy for params={(learn_rate, depth)}{' with scaler ' if scale else ''} is {acc * 100}%")


def fine_tune_light_gbm():
    data = get_data()
    kf = KFold()
    for scale in [True, False]:
        acc = 0
        for train_index, val_index in kf.split(data.train_x):
            gbm = LGBMClassifier(random_state=42)
            train_x, train_y = _get_kfold(data, train_index)
            val_x, val_y = _get_kfold(data, val_index)
            if scale:
                train_x, val_x = _scale(train_x, val_x)
            gbm.fit(train_x, train_y)
            acc += accuracy_score(val_y, gbm.predict(val_x))
        acc /= N_SPLITS
        print(
            f"Light GBM: Accuracy {'with scaler ' if scale else ''} is {acc * 100}%")


if __name__ == '__main__':
    fine_tune_knn()
    fine_tune_forest()
    vanila_forest()
    fine_tune_gb()
    fine_tune_light_gbm()
