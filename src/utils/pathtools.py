import collections
import datetime
import os
from pathlib import Path
import typing as t


class CustomizedPath():

    def __init__(self):
        self._root = Path(__file__).parent.parent.parent

        # Logs initialized
        self._initialized_loggers = collections.defaultdict(bool)

        # Datasets promises
        self._train = None
        self._test = None
        self._sample = None

# ------------------ UTILS ------------------

    def remove_prefix(input_string: str, prefix: str) -> str:
        """Removes the prefix if exists at the beginning in the input string
        Needed for Python<3.9
        
        :param input_string: The input string
        :param prefix: The prefix
        :returns: The string without the prefix
        """
        if prefix and input_string.startswith(prefix):
            return input_string[len(prefix):]
        return input_string

    def as_relative(self, path: t.Union[str, Path]) -> Path:
        """Removes the prefix `self.root` from an absolute path.

        :param path: The absolute path
        :returns: A relative path starting at `self.root`
        """
        if type(path) == str:
            path = Path(path)
        return Path(CustomizedPath.remove_prefix(path.as_posix(), self.root.as_posix()))

    def mkdir_if_not_exists(self, path: Path, gitignore: bool=False) -> Path:
        """Makes the directory if it does not exists

        :param path: The input path
        :param gitignore: A boolean indicating if a gitignore must be included for the content of the directory
        :returns: The same path
        """
        path.mkdir(parents=True, exist_ok = True)

        if gitignore:
            with (path / '.gitignore').open('w') as f:
                f.write('*\n!.gitignore')

        return path

# ------------------ MAIN FOLDERS ------------------

    @property
    def root(self):
        return self._root

    @property
    def data(self):
        return self.mkdir_if_not_exists(self.root / 'data', gitignore=True)

    @property
    def output(self):
        return self.mkdir_if_not_exists(self.root / 'output', gitignore=True)

    @property
    def logs(self):
        return self.mkdir_if_not_exists(self.root / 'logs', gitignore=True)

# ------------------ DOWNLOADED DATA ------------------

    def check_downloaded(self, file_name: str) -> Path:
        """Checks that the file was correctly downloaded by the user.

        :param file_name: The file to check, that should be in `data/`
        :returns: A Path object to this file.
        """
        result = self.data / file_name
        if not os.path.exists(result):
            raise FileNotFoundError('You should first download the data. Please refer to the instruction in README.md')

        return result

    @property
    def edge_attributes(self) -> Path:
        return self.check_downloaded('edge_attributes.txt')

    @property
    def edgelist(self) -> Path:
        return self.check_downloaded('edgelist.txt')

    @property
    def graph_indicator(self) -> Path:
        return self.check_downloaded('graph_indicator.txt')

    @property
    def graph_labels(self) -> Path:
        return self.check_downloaded('graph_labels.txt')

    @property
    def node_attributes(self) -> Path:
        return self.check_downloaded('node_attributes.txt')

    @property
    def sequences(self) -> Path:
        return self.check_downloaded('sequences.txt')


# ------------------ LOGS ------------------

    def get_log_file(self, logger_name: str) -> Path:
        """Creates and initializes a logger.

        :param logger_name: The logger name to create
        :returns: A path to the `logger_name.log` created and/or initialized file
        """
        file_name = logger_name + '.log'
        result = self.logs / file_name

        # Checking if exists
        if not os.path.isfile(result):
            with result.open('w') as f:
                pass

        # Header for new log
        if not self._initialized_loggers[logger_name]:
            with result.open('a') as f:
                f.write(f'\nNEW LOG AT {datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")}\n')
            self._initialized_loggers[logger_name] = True

        return result

# ------------------ EMBEDDINGS ------------------

    @property
    def embeddings(self):
        return self.mkdir_if_not_exists(self.output / 'embeddings')

    def get_embedding_folder(self, embedding_name):
        """Returns the embedding folder for a given class of embedding.
        Ex of class of embedding: `'prot_bert'`

        :param embedding_name: The class of embedding
        :returns: The path to the folder
        """
        return self.mkdir_if_not_exists(self.embeddings / embedding_name)

    def get_new_embedding_file(self, embedding_name: str):
        """Returns a new embedding file for a given class.
        Ex of class of embedding: `'prot_bert'`

        :param embedding_name: The class of embedding
        :returns: The path to the new file
        """
        parent_folder = self.get_embedding_folder(embedding_name)
        result = parent_folder / f'{embedding_name}_{datetime.datetime.now().strftime("embeddings_%Y_%m%d__%H_%M_%S")}.csv'
        with result.open('w') as f:
            pass
        return result

    def get_latest_embeddings(self, embedding_name: str):
        """Returns the latest embedding file for a given class.
        Ex of class of embedding: `'prot_bert'`

        :param embedding_name: The class of embedding
        :returns: The path to the embedding file
        """
        parent_folder = self.get_embedding_folder(embedding_name)
        files = sorted([
            str(path)
            for path in parent_folder.iterdir()
            if path.is_file()
        ])

        if len(files) == 0:
            return None
        return Path(files[-1])

# ------------------ FEATURES ------------------

    @property
    def features(self):
        return self.mkdir_if_not_exists(self.output / 'features')

    def get_feature_folder(self, feature_name):
        """Returns the feature folder for a given class of feature.
        Ex of class of feature: `'hgp'`

        :param feature_name: The class of feature
        :returns: The path to the folder
        """
        return self.mkdir_if_not_exists(self.features / feature_name)

    def get_new_feature_file(self, feature_name: str):
        """Returns a new feature file for a given class.
        Ex of class of feature: `'hgp'`

        :param feature_name: The class of feature
        :returns: The path to the new file
        """
        parent_folder = self.get_feature_folder(feature_name)
        result = parent_folder / f'{feature_name}_{datetime.datetime.now().strftime("features_%Y_%m%d__%H_%M_%S")}.csv'
        with result.open('w') as f:
            pass
        return result

    def get_latest_features(self, feature_name: str):
        """Returns the latest feature file for a given class.
        Ex of class of feature: `'hgp'`

        :param feature_name: The class of feature
        :returns: The path to the feature file
        """
        parent_folder = self.get_feature_folder(feature_name)
        files = sorted([
            str(path)
            for path in parent_folder.iterdir()
            if path.is_file()
        ])

        if len(files) == 0:
            return None
        return Path(files[-1])

# ------------------ SUBMISSIONS ------------------
    
    @property
    def submission_folder(self):
        return self.mkdir_if_not_exists(self.output / 'submissions')
    
    def get_new_submission_file(self):
        """Returns a new submission file.

        :returns: The path to the new file
        """
        result = self.submission_folder / f'submission_{datetime.datetime.now().strftime("%Y_%m%d__%H_%M_%S")}.csv'
        with result.open('w') as f:
            pass
        return result

project = CustomizedPath() 