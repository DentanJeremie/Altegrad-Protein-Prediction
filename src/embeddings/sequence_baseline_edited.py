import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

from src.utils.constants import SEQUENCE_BASELINE_FEATURE_NAME as FEATURE_NAME
from src.utils.pathtools import project
from src.utils.logging import logger
from src.utils.sequence_data import SequenceData, sequence_data
from src.utils.constants import *

VECTORIZER_ANALYSER = 'char'
VECTORIZER_NGRAMS_RANGE = (1, 4)
N_COMPONENT = 32


class SequenceBaseline():

    def __init__(self, data: SequenceData = sequence_data):
        self.data: SequenceData = data

# ------------------ BUILDS ------------------

    def compute_embeddings(self):
        """Does the vectorization of the sequences, with TFIDF.
        Builds the attributes:
        * `self._x_train`
        * `self._x_test`
        """
        logger.info('Computing TF-IDF vectorization...')
        vec = TfidfVectorizer(analyzer=VECTORIZER_ANALYSER, ngram_range=VECTORIZER_NGRAMS_RANGE)
        self.embeddings = vec.fit_transform(self.data.sequences)
        
        logger.info('Computing SVD of the vectors...')
        svd = TruncatedSVD(n_components=N_COMPONENT)
        self.embeddings = svd.fit_transform(self.embeddings)

        output_path = project.get_new_embedding_file(TFIDF_EMBEDDING)
        pd.DataFrame(self.embeddings).to_csv(output_path, index=False)


def main():
    sequence_baseline = SequenceBaseline()
    sequence_baseline.compute_embeddings()

if __name__ == '__main__':
    main()