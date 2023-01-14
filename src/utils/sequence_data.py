import typing as t
import csv

from src.utils.pathtools import project
from src.utils.logging import logger

class SequenceData():

    """
    Class representing the sequence data of the proteins.
    
    The attributes of this class are:
        * `self.sequences`: list of the sequences
        * `self.sequences_train`: list of the sequences of the train set
        * `self.sequences_test`: list of the sequences of the test set
        * `self.proteins_test`: IDs of the proteins of the test set
        * `self.y_train`: labels of the proteins of the train set
    """

    def __init__(self):
        self._sequences: t.List[str] = None
        self._sequences_train: t.List[str] = None
        self._sequences_test: t.List[str] = None
        self._proteins_test: t.List[str] = None
        self._y_train: t.List[str] = None

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
    def sequences_test(self):
        if self._sequences_test is None:
            self.build()
        return self._sequences_test

    @property
    def proteins_test(self):
        if self._proteins_test is None:
            self.build()
        return self._proteins_test

    @property
    def y_train(self):
        if self._y_train is None:
            self.build()
        return self._y_train

# ------------------ BUILD ------------------

    def build(self):
        """Builds the attributes of the object:

        * `self._sequences`
        * `self._sequences_train`
        * `self._sequences_test`
        * `self._proteins_test`
        * `self._y_train`
        """
        # Read sequences
        self._sequences = list()
        with project.sequences.open('r') as f:
            for line in f:
                self._sequences.append(line[:-1])

        # Split data into training and test sets
        self._sequences_train = list()
        self._sequences_test = list()
        self._proteins_test = list()
        self._y_train = list()
        with project.graph_labels.open('r') as f:
            for i,line in enumerate(f):
                t = line.split(',')
                if len(t[1][:-1]) == 0:
                    self._proteins_test.append(t[0])
                    self._sequences_test.append(self._sequences[i])
                else:
                    self._sequences_train.append(self._sequences[i])
                    self._y_train.append(int(t[1][:-1]))

sequence_data = SequenceData()
