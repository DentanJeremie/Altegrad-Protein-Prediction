import csv
import json
import os
import typing as t

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import xgboost

from src.utils.constants import *
from src.utils.logging import logger
from src.utils.pathtools import project
from src.utils.train_validation_test import SetsManager, sets_manager

INDEX_COLUMN = 'Unnamed: 0'


class FinalClassifier(object):

    def __init__(self, sets: SetsManager):
        self.sets = sets
        self._trained = False

        self.load_features()
        self.init_classifier(tune_xgb=True)

        logger.info('All features have been loaded, the classifier is initialized.')
    
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
                project.get_latest_features(feature),
                low_memory=False,
            ).drop(INDEX_COLUMN, axis=1)


        # Labels
        self.full_train_labels = project.train[DATA_LABEL].to_frame()
        logger.info(len(self.full_train_labels.index))

    def split_train_test(self, test_size: float):
        """Splits the full_train set into train and test.
        :param test_size: The proportion of sample in the test set.
        """
        logger.info('Splitting into train and test set')
        (
            self.train_features,
            self.test_features,
            self.train_labels,
            self.test_labels,
        ) = train_test_split(
            self.full_train_features,
            self.full_train_labels,
            test_size=test_size
        )
        
    def init_classifier(self, tune_xgb = False, force_default = False):
        """Initiates the XGBoost classifier.
        """
        logger.info('Starting XGB tuning')
        self.dtrain = xgboost.DMatrix(self.train_features, label = self.train_labels)
        self.dtest = xgboost.DMatrix(self.test_features, label = self.test_labels)
        self.dsubmission = xgboost.DMatrix(self.submission_features)

        if tune_xgb:
            searched_parameters = self.xgb_tuning()
        else:
            searched_parameters = XGB_DEFAULT_PARAM_TO_SEARCH
            if not force_default and os.path.isfile(project.xgboost_parameters):
                try:
                    searched_parameters = json.loads(str(project.xgboost_parameters))
                except FileNotFoundError:
                    pass

        self.xgb_params = {**searched_parameters, **XGB_ADDITIONNAL_PARAM}

    def xgb_tuning(self):
        xgbc = xgboost.XGBClassifier(objective='multi:softmax', num_class=3)
        clf = GridSearchCV(estimator=xgbc, 
            param_grid=XGB_PARAM_SEARCH,
            scoring='accuracy', 
            verbose=1
        )
        clf.fit(self.full_train_features, self.full_train_labels)
        result = clf.best_params_
        logger.info('End of XGB tuning')

        # Saving
        with project.xgboost_parameters.open('w') as f:
            json.dump(result, f)
        logger.info(f'Tuning parameters stored at {project.as_relative(project.xgboost_parameters)}')

        return result

    def train(self):
        logger.info('Training XGBoost classifier')
        self.trained_model = xgboost.train(
            self.xgb_params,
            self.dtrain,
        )

        self._trained = True

    def eval(self):
        logger.info('Evaluating the model')
        test_predictions = self.trained_model.predict(self.dtest)
        test_predictions = test_predictions.round(0).astype(int)
        test_labels = np.array(self.test_labels[DATA_LABEL].values)

        # Metrics
        evaluation_results = {
            'accuracy':accuracy_score(y_true=test_labels, y_pred=test_predictions),
            'recall':recall_score(y_true=test_labels, y_pred=test_predictions, average='weighted', zero_division=0.0),
            'precision':precision_score(y_true=test_labels, y_pred=test_predictions, average='weighted', zero_division=0.0),
            'f1':f1_score(y_true=test_labels, y_pred=test_predictions, average='weighted', zero_division=0.0),
        }
        for metric in evaluation_results:
            logger.info(f'XGBoost evaluation: {metric}: {evaluation_results[metric]}')

    def predict(self, eval_first = True):
        """Makes the final prediction
        """
        if not self._trained:
            self.train()

        if eval_first:
            self.eval()

        # Loading the pair labels
        test_ids = project.test[DATA_ID].values

        logger.info('Computing predictions for the submission')
        submission_predictions = self.trained_model.predict(self.dsubmission)
        submission_predictions = submission_predictions.round(0).astype(int)

        destination = project.get_new_submission_file()
        with destination.open('w') as f:
            csv_out = csv.writer(f, lineterminator='\n')
            csv_out.writerow(['id','prediction'])
            for i, row in zip(test_ids, submission_predictions):
                csv_out.writerow([i, row])

        logger.info(f'Submission stored at {project.as_relative(destination)}')

def main():
    final_xgboost = FinalClassifier()
    final_xgboost.predict()

if __name__ == '__main__':
    main()