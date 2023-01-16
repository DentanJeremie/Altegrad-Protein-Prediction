import typing as t
import csv

from src.utils.pathtools import project
from src.utils.logging import logger
from src.utils.train_validation_test import sets_manager

class SequenceData():

    """
    Class representing the sequence data of the proteins.
    
    The attributes of this class are:
        * `self.sequences`: list of the sequences
        * `self.sequences_train`: list of the sequences of the train set
        * `self.y_train`: labels of the proteins of the train set
        * `self.sequences_validation`: list of the sequences of the validation set
        * `self.y_validation`: labels of the proteins of the validation set
        * `self.sequences_test`: list of the sequences of the test set
        * `self.proteins_test`: IDs of the proteins of the test set

    For the definition of train/validation/test sets, please refer to `src.utils.train_validation_test`.
    """

    def __init__(self):
        self._sequences: t.List[str] = None
        self._sequences_train: t.List[str] = None
        self._y_train: t.List[str] = None
        self._sequences_validation: t.List[str] = None
        self._y_validation: t.List[str] = None
        self._sequences_test: t.List[str] = None
        self._proteins_test: t.List[str] = None

# ------------------ PROPERTIES ------------------

    @property
    def sequences(self):
        if self._sequences is None:
            self.build()
        return self._sequences
    
    @property
    def sequences_train(self):
        if self._sequences_train is None:
            self.build()
        return self._sequences_train

    @property
    def y_train(self):
        if self._y_train is None:
            self.build()
        return self._y_train

    @property
    def sequences_validation(self):
        if self._sequences_validation is None:
            self.build()
        return self._sequences_validation

    @property
    def y_validation(self):
        if self._y_validation is None:
            self.build()
        return self._y_validation

    @property
    def sequences_test(self):
        if self._sequences_test is None:
            self.build()
        return self._sequences_test

    @property
    def proteins_test(self):
        if self._proteins_test is None:
            self.build()
        return self._proteins_test

# ------------------ BUILD ------------------

    def build(self):
        """Builds the attributes of the object:

        * `self._sequences`
        * `self._sequences_train`
        * `self._y_train`
        * `self._sequences_validation`
        * `self._y_validation`
        * `self._sequences_test`
        * `self._proteins_test`
        """
        # Read sequences
        self._sequences = list()
        with project.sequences.open('r') as f:
            for line in f:
                self._sequences.append(line[:-1])

        # Split data into training and test sets
        self._sequences_train = list()
        self._y_train = list()
        self._sequences_validation = list()
        self._y_validation = list()
        self._sequences_test = list()
        self._proteins_test = list()

        for index, seq in enumerate(self._sequences):
            if sets_manager.is_train(index):
                self._sequences_train.append(seq)
                self._y_train.append(sets_manager.get_label(index))
            if sets_manager.is_validation(index):
                self._sequences_validation.append(seq)
                self._y_validation.append(sets_manager.get_label(index))
            if sets_manager.is_test(index):
                self._proteins_test.append(sets_manager.index_to_protein(index))
                self._sequences_test.append(seq)

sequence_data = SequenceData()
