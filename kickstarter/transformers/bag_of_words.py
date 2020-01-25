import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from kickstarter.transformers._base.base_transformer import BaseTransformer

BATCH_SIZE = 10000

STOP_WORDS = set(stopwords.words('english'))


class BagOfWords(BaseTransformer):
    def __init__(self) -> None:
        self._vectorizer = CountVectorizer(min_df=0.0001, stop_words=STOP_WORDS)
        self._true_proba = None  # P(y=1|x)
        self._false_proba = None  # P(y=0|x

    @property
    def input_fields(self) -> str:
        return "blurb"

    def fit(self, x: np.ndarray, y: np.ndarray):
        self._vectorizer.fit(x)
        x_true = x[y == 1]
        x_false = x[y == 0]

        true_counts = np.sum(self._vectorizer.transform(x_true), axis=0) + 1
        false_count = np.sum(self._vectorizer.transform(x_false), axis=0) + 1

        self._true_proba = true_counts / (true_counts + false_count)  # #{y=1.x} / #{x}
        self._false_proba = false_count / (true_counts + false_count)  # #{y=0,x} / #{x}

    def transform(self, x: pd.Series) -> pd.DataFrame:
        result = []
        for i in range(0, len(x) + BATCH_SIZE - 1, BATCH_SIZE):
            transformed_x = self._vectorizer.transform(x[i:i + BATCH_SIZE]).todense()

            true_prob = self._get_proba(transformed_x, self._true_proba)
            false_prob = self._get_proba(transformed_x, self._false_proba)

            result.extend(np.array(true_prob > false_prob, dtype="int"))

        return pd.DataFrame(result, index=x.index, columns=["bag_of_words"])

    @staticmethod
    def _get_proba(transformed_x: np.ndarray, proba_vector: np.ndarray) -> np.ndarray:
        # P(y| #{x_i} ) = P(y | x_i)^#{x_i}
        prob = np.power(proba_vector, transformed_x)

        # P(y | x_1, ..., x_n) = multiply for i ( P(y | x_i)^#{x_i} )
        prob = np.prod(prob, axis=1)

        return np.reshape(np.where(prob == 1, 0, prob), len(prob))


if __name__ == '__main__':
    from sklearn.feature_extraction.text import TfidfVectorizer
    from nltk.corpus import stopwords
    from sklearn.model_selection import train_test_split
    from lightgbm import LGBMClassifier

    from sklearn.preprocessing import StandardScaler

    bag = BagOfWords()
    input_fields = ['launched_at_month', 'launched_at_year', 'category', 'parent_category', 'destination_delta_in_days',
                    'goal', 'blurb']
    input_fields += ['nima_score', 'nima_tech']
    cols = list(df.columns.values)
    fields = [field for field in input_fields if field in cols]
    X = df[fields]
    y = np.array(df['state'] == "successful", dtype="int")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

    s = set(stopwords.words('english'))
    X_train = X_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    vec = TfidfVectorizer(min_df=0.01, stop_words=s)

    X = vec.fit_transform(X_train["blurb"].astype("str"))
    added_tfidf = pd.DataFrame(X.toarray())
    added_bag = bag.fit_transform(X_train["blurb"].astype("str"), y_train)

    X_train = pd.concat([X_train, added_tfidf, added_bag], axis=1)
    X_train = X_train.drop(columns='blurb')

    X = vec.transform(X_test["blurb"].astype("str"))
    added_tfidf = pd.DataFrame(X.toarray())
    added_bag = bag.transform(X_test["blurb"].astype("str"))

    X_test = pd.concat([X_test, added_tfidf, added_bag], axis=1)
    X_test = X_test.drop(columns='blurb')

    scaler = StandardScaler()
    scaler.fit(X_train)

    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    error = []
    forest = LGBMClassifier()
    forest.fit(X_train, y_train)
    pred = forest.predict(X_test)
    print('precision is: ' + str(1 - np.mean(pred != y_test)))
    print(1 - np.mean(pred != y_test))
