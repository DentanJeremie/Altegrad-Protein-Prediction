"""
Script used to manage the TRAIN-VALIDATION-TEST sets of the project.
* Both the TRAIN and VALIDATION sets are annotated
* The TEST set is the set of proteins for which we do not have annotation (test on Kaggle)
* We must have the same TRAIN and VALIDATION set for all our models to ensure consistency of the evaluation
"""

import typing as t

from numpy.random import RandomState

from src.utils.pathtools import project
from src.utils.logging import logger

PROP_VALIDATION = 0.05
SEED = 2

class SetsManager():

    def __init__(self):
        """Creates the three attributes `self.train`, `self.validation`, and `self.test`.
        Each of those attributes is a list whose elements are dicts {'protein_id':prot_id, 'index':prot_index}
        for the corresponding sets.
        """
        # Private attributes with properties
        self._full_train: t.List[t.Dict] = None
        self._train: t.List[t.Dict] = None
        self._validation: t.List[t.Dict] = None
        self._test: t.List[t.Dict] = None
        self._train_ids: t.List[str] = None
        self._train_indexes: t.List[int] = None
        self._validation_ids: t.List[str] = None
        self._validation_indexes: t.List[int] = None
        self._test_ids: t.List[str] = None
        self._test_indexes: t.List[int] = None
        self._labels: t.Dict[t.Union[str, int], int] = None

        # Private attributes without properties
        self._idx_to_ids: t.Dict[int, str] = None
        self._ids_to_idx: t.Dict[str, int] = None

# ------------------ PROPERTIES ------------------

    @property
    def full_train(self):
        if self._full_train is None:
            self.build()
        return self._full_train

    @property
    def train(self):
        if self._train is None:
            self.build()
        return self._train

    @property
    def validation(self):
        if self._validation is None:
            self.build()
        return self._validation

    @property
    def test(self):
        if self._test is None:
            self.build()
        return self._test

    @property
    def train_ids(self):
        if self._train_ids is None:
            self.build()
        return self._train_ids

    @property
    def train_indexes(self):
        if self._train_indexes is None:
            self.build()
        return self._train_indexes

    @property
    def validation_ids(self):
        if self._validation_ids is None:
            self.build()
        return self._validation_ids

    @property
    def validation_indexes(self):
        if self._validation_indexes is None:
            self.build()
        return self._validation_indexes

    @property
    def test_ids(self):
        if self._test_ids is None:
            self.build()
        return self._test_ids

    @property
    def test_indexes(self):
        if self._test_indexes is None:
            self.build()
        return self._test_indexes

    @property
    def labels(self):
        if self._labels is None:
            self.build()
        return self._labels

# ------------------ BUILDS ------------------
    
    def build(self):
        """Builds the main attributes of the object.
        """
        logger.info('Building a deterministic separation of train/validations/test sets...')
        # Data
        random_state = RandomState(SEED)
        self.load_test_full_train()

        # Random choices
        num_full_train = len(self._full_train)
        num_validation = int(PROP_VALIDATION * num_full_train)
        logger.debug(f'Found {num_full_train} proteins in full train, choosing {num_validation} for validation...')
        validation_indexes = random_state.choice(
            num_full_train,
            num_validation,
            replace=False,
        )

        # Actual lists
        self._train = list()
        self._validation = list()
        for index, elt in enumerate(self.full_train):
            if index in validation_indexes:
                self._validation.append(elt)
            else:
                self._train.append(elt)

        # Flatten lists
        self._train_ids = [elt['protein_id'] for elt in self._train]
        self._train_indexes = [elt['index'] for elt in self._train]
        self._validation_ids = [elt['protein_id'] for elt in self._validation]
        self._validation_indexes = [elt['index'] for elt in self._validation]
        self._test_ids = [elt['protein_id'] for elt in self._test]
        self._test_indexes = [elt['index'] for elt in self._test]

        # Inverse dicts
        self._ids_to_idx = dict()
        self._idx_to_ids = dict()
        for elt in self._full_train:
            self._idx_to_ids[elt['index']] = elt['protein_id']
            self._ids_to_idx[elt['protein_id']] = elt['index']
        for elt in self.test:
            self._idx_to_ids[elt['index']] = elt['protein_id']
            self._ids_to_idx[elt['protein_id']] = elt['index']

    def load_test_full_train(self):
        """Initiates `self._test` and `self._full_train`.
        The `full_train` set corresponds to all proteins for which we have annotations
        It is not the same as `train`, which is a subset of `full_train` that does not contain
        the elements of the validation set.
        """
        self._test = list()
        self._full_train = list()
        self._labels = dict()

        logger.debug('Reading the data to separate full_train and test sets...')
        with project.graph_labels.open('r') as f:
            for i,line in enumerate(f):
                t = line.split(',')
                if len(t[1][:-1]) == 0:
                    self._test.append({
                        'protein_id':t[0],
                        'index':i,
                    })
                else:
                    self._full_train.append({
                        'protein_id':t[0],
                        'index':i,
                    })
                    self._labels[t[0]] = int(t[1][:-1])
                    self._labels[i] = int(t[1][:-1])

# ------------------ METHODS ------------------

    def is_train(self, id: t.Union[str, int]) -> bool:
        """Given either a protein index or a protein ID, decides whether 
        it is part of the train set or not
        
        :param id: Either the protein index of it ID
        :return: Boolean describing if it is part of the train set.
        """
        if type(id) == str:
            return id in self.train_ids
        if type(id) == int:
            return id in self.train_indexes

        raise ValueError(f'Incorrect ID type provided: {id}')

    def is_validation(self, id: t.Union[str, int]) -> bool:
        """Given either a protein index or a protein ID, decides whether 
        it is part of the validation set or not
        
        :param id: Either the protein index of it ID
        :return: Boolean describing if it is part of the validation set.
        """
        if type(id) == str:
            return id in self.validation_ids
        if type(id) == int:
            return id in self.validation_indexes
            
        raise ValueError('Incorrect ID type provided.')

    def is_test(self, id: t.Union[str, int]) -> bool:
        """Given either a protein index or a protein ID, decides whether 
        it is part of the test set or not
        
        :param id: Either the protein index of it ID
        :return: Boolean describing if it is part of the test set.
        """
        if type(id) == str:
            return id in self.test_ids
        if type(id) == int:
            return id in self.test_indexes
            
        raise ValueError('Incorrect ID type provided.')

    def get_label(self, id: t.Union[str, int]) -> int:
        """Given either a protein index or a protein ID, returns its label
        if it's in the train or validation set.
        
        :param id: Either the protein index of it ID
        :return: The label
        """
        if id in self.labels:
            return self.labels[id]
        raise IndexError('You asked for the label of a protein that is not in the train nor the validation set.')

    def protein_to_index(self, id: str) -> int:
        """Given the ID of a protein, returns its index.
        
        :param id: The ID of the proteins
        :returns: Its index
        """
        if self._ids_to_idx is None:
            self.build()
        
        return self._ids_to_idx[id]

    def index_to_protein(self, id: str) -> int:
        """Given the index of a protein, returns its ID.
        
        :param id: The index of the proteins
        :returns: Its ID
        """
        if self._idx_to_ids is None:
            self.build()
        
        return self._idx_to_ids[id]

sets_manager = SetsManager()
