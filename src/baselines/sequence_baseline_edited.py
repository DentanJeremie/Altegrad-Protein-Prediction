import csv
import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from src.utils.constants import SEQUENCE_BASELINE_FEATURE_NAME as FEATURE_NAME
from src.utils.pathtools import project
from src.utils.logging import logger
from src.utils.sequence_data import SequenceData, sequence_data

VECTORIZER_ANALYSER = 'char'
VECTORIZER_NGRAMS_RANGE = (1, 3)
LOGIT_SOLVER = 'liblinear'


class SequenceBaseline():

    def __init__(self, data: SequenceData = sequence_data):
        self.data: SequenceData = data
        self._x_train: sp.csr_matrix = None
        self._x_test: sp.csr_matrix = None
        self._y_pred_proba: np.ndarray = None

# ------------------ PROPERTIES ------------------

    @property
    def x_train(self):
        if self._x_train is None:
            self.do_vectorization()
        return self._x_train

    @property
    def x_test(self):
        if self._x_test is None:
            self.do_vectorization()
        return self._x_test

    @property
    def y_pred_proba(self):
        if self._y_pred_proba is None:
            self.do_logit_regression()
        return self._y_pred_proba

# ------------------ BUILDS ------------------

    def do_vectorization(self):
        """Does the vectorization of the sequences, with TFIDF.
        Builds the attributes:
        * `self._x_train`
        * `self._x_test`
        """
        vec = TfidfVectorizer(analyzer=VECTORIZER_ANALYSER, ngram_range=VECTORIZER_NGRAMS_RANGE)
        self._x_train = vec.fit_transform(self.data.sequences_train)
        self._x_test = vec.transform(self.data.sequences_test)

    def do_logit_regression(self):
        """Does the logistic regression for the prediction.
        Builds the attributes:
        * `self._y_pred_proba`
        """
        clf = LogisticRegression(solver=LOGIT_SOLVER)
        clf.fit(self.x_train, self.data.y_train)
        self._y_pred_proba = clf.predict_proba(self.x_test)

# ------------------ PREDICTION ------------------

    def predict(self):
        """Does the prediction and saves it.
        """
        with project.get_new_feature_file(FEATURE_NAME).open('w') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            lst = list()
            for i in range(18):
                lst.append('class'+str(i))
            lst.insert(0, "name")
            writer.writerow(lst)
            for i, protein in enumerate(self.data.proteins_test):
                lst = self.y_pred_proba[i,:].tolist()
                lst.insert(0, protein)
                writer.writerow(lst)

def main():
    sequence_baseline = SequenceBaseline()
    sequence_baseline.predict()

if __name__ == '__main__':
    main()