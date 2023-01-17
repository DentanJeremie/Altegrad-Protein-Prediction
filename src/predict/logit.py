import csv
import json
import os
import typing as t

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, log_loss
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
import xgboost

from src.utils.constants import *
from src.utils.logging import logger
from src.utils.pathtools import project
from src.utils.sequence_data import sequence_data, SequenceData
from src.utils.train_validation_test import SetsManager, sets_manager

HEADER = (
    'name,class0,class1,class2,class3,class4,class5,class6,'
    'class7,class8,class9,class10,class11,class12,class13,'
    'class14,class15,class16,class17'
)
LOGIT_SOLVER = 'liblinear'


class FinalClassifier(object):

    def __init__(self, sets: SetsManager = sets_manager, data: SequenceData = sequence_data):
        self.sets = sets
        self.data = data
        self._trained = False
    
    def load_features(self):
        """Loads the features to the classifier.
        """
        # Initialization
        full_features: t.Dict[str, pd.DataFrame] = dict()

        # Loading features
        logger.info('Loading high-level features')
        for feature in EMBEDDINGS:
            logger.debug(f'Loading features for {feature}...')

            full_features[feature] = pd.read_csv(
                project.get_latest_embeddings(feature),
                low_memory=False,
            )

         # Concatenating
        logger.info('Concatenating loaded high-level features')
        self.full_features: pd.DataFrame = pd.concat([
            full_features[feature].add_prefix(f'{feature}-')
            for feature in EMBEDDINGS
        ], axis = 1)

    def split_train_test(self):
        """Splits the full_train set into train and test.
        """
        logger.info('Splitting into train validation test sets')

        # Features
        self.train_features = self.full_features.loc[self.sets.train_indexes]
        self.validation_features = self.full_features.loc[self.sets.validation_indexes]
        self.test_features = self.full_features.loc[self.sets.test_indexes]

        # Labels
        self.train_labels = [
            self.sets.get_label(index)
            for index in self.sets.train_indexes
        ]
        self.validation_labels = [
            self.sets.get_label(index)
            for index in self.sets.validation_indexes
        ]

        # Label encoding for the labels
        le = LabelEncoder()
        self.train_labels = le.fit_transform(self.train_labels)
        self.validation_labels = le.fit_transform(self.validation_labels)

    def train(self):
        """TO BE COMPLETED"""
        self.clf = LogisticRegression(solver=LOGIT_SOLVER)
        self.clf.fit(self.train_features, self.data.y_train)
    
    def eval(self):
        """TO BE COMPLETED"""
        y_preds_validation = self.clf.predict_proba(self.validation_features)
        logger.info(f'log_loss : {log_loss(y_true=self.data.y_validation, y_pred=y_preds_validation)}')
    
    def predict(self, eval_first = True):
        """Makes the final prediction
        """

        # Loading the pair labels
        test_ids = [
            self.sets.index_to_protein(index)
            for index in self.sets.test_indexes
        ]

        logger.info('Computing predictions for the submission')
        test_predictions = self.clf.predict_proba(self.test_features)

        destination = project.get_new_submission_file()
        with destination.open('w') as f:
            csv_out = csv.writer(f, lineterminator='\n')
            csv_out.writerow(HEADER.split(','))
            for i, row in zip(test_ids, test_predictions):
                csv_out.writerow([i, *row])

        logger.info(f'Submission stored at {project.as_relative(destination)}')

def main():
    final_clf = FinalClassifier()
    final_clf.load_features()
    final_clf.split_train_test()
    final_clf.train()
    final_clf.eval()
    final_clf.predict()

if __name__ == '__main__':
    main()