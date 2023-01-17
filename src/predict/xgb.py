import csv
import json
import os
import typing as t

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, log_loss
from sklearn.preprocessing import LabelEncoder
import xgboost

from src.utils.constants import *
from src.utils.logging import logger
from src.utils.pathtools import project
from src.utils.train_validation_test import SetsManager, sets_manager

XGB_PARAM_SEARCH = {
    'max_depth': [2, 3, 4],
    'learning_rate': [0.01, 0.02, 0.05, 0.1, 0.2],
    'colsample_bytree': [0.3, 0.2, 0.5, 0.8],
    'min_child_weight': [0.5, 1, 2],
}
XGB_DEFAULT_PARAM_TO_SEARCH = {
    'max_depth': 5,
    'learning_rate': 0.1,
    'colsample_bytree': 0.8,
    'min_child_weight': 2,
}
XGB_ADDITIONNAL_PARAM = {
    'objective': 'multi:softmax',
    'num_class': 18,
    'eta': 0.3,
    'subsample': 0.5,
    'gamma': 1,
    'eval_metric': 'logloss',
}
HEADER = (
    'name,class0,class1,class2,class3,class4,class5,class6,'
    'class7,class8,class9,class10,class11,class12,class13,'
    'class14,class15,class16,class17'
)


class FinalClassifier(object):

    def __init__(self, sets: SetsManager = sets_manager):
        self.sets = sets
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
        
        print(self.full_features)

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

        
    def init_classifier(self, tune_xgb = False, force_default = False):
        """Initiates the XGBoost classifier.
        """
        logger.info('Starting XGB tuning')
        self.dtrain = xgboost.DMatrix(self.train_features, label = self.train_labels)
        self.dvalidation = xgboost.DMatrix(self.validation_features, label = self.validation_labels)
        self.dtest = xgboost.DMatrix(self.test_features)

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
            scoring='neg_log_loss', 
            verbose=1
        )
        clf.fit(self.train_features, self.train_labels)
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
        validation_predictions = self.trained_model.predict_proba(self.dvalidation)
        validation_labels = np.array(self.validation_labels)

        # Metrics
        evaluation_results = {
            'log_loss':log_loss(y_true=validation_labels, y_pred=validation_predictions),
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
        test_ids = [
            self.sets.index_to_protein(index)
            for index in self.sets.test_indexes
        ]

        logger.info('Computing predictions for the submission')
        test_predictions = self.trained_model.predict(self.dtest)

        destination = project.get_new_submission_file()
        with destination.open('w') as f:
            csv_out = csv.writer(f, lineterminator='\n')
            csv_out.writerow(HEADER.split(','))
            for i, row in zip(test_ids, test_predictions):
                csv_out.writerow([i, row])

        logger.info(f'Submission stored at {project.as_relative(destination)}')

def main():
    final_xgboost = FinalClassifier()
    final_xgboost.load_features()
    final_xgboost.split_train_test()
    final_xgboost.init_classifier()
    final_xgboost.predict()

if __name__ == '__main__':
    main()