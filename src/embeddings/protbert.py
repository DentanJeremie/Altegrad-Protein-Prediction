"""
This scripts implements the computation of embeddings from the sequences.
It uses a pretrained model, and is inspired from a code taken in https://github.com/agemagician/ProtTrans
"""

import typing as t

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import torch
from transformers import AlbertTokenizer, XLNetTokenizer, AutoTokenizer, AutoModel, pipeline

from src.utils.constants import *
from src.utils.pathtools import project
from src.utils.logging import logger
from src.utils.sequence_data import SequenceData, sequence_data
from src.utils.train_validation_test import SetsManager, sets_manager

TASK = 'feature-extraction'
BERT_MODEL = 'Rostlab/prot_bert'
ALBERT_MODEL = 'Rostlab/prot_albert'
BERT_BFD_MODEL = 'Rostlab/prot_bert_bfd'
XLNET_MODEL = 'Rostlab/prot_xlnet'
N_COMPONENT_PCA = 32


class ProtBertClassifier():

    def __init__(
        self,
        data: SequenceData = sequence_data,
        sets: SetsManager = sets_manager,
    ):
        # Data
        self.data = data
        self.sets = sets
        
        # Building models...
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        # BERT
        self.bert_model = AutoModel.from_pretrained(BERT_MODEL)
        self.bert_tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL, do_lower_case=False)
        self.bert_pipeline = pipeline(TASK, model = self.bert_model, tokenizer=self.bert_tokenizer, device=self.device)
        # ALBERT
        self.albert_model = AutoModel.from_pretrained(ALBERT_MODEL)
        self.albert_tokenizer = AlbertTokenizer.from_pretrained(ALBERT_MODEL, do_lower_case=False)
        self.albert_pipeline = pipeline(TASK, model = self.albert_model, tokenizer=self.albert_tokenizer, device=self.device)
        # BERT BFD
        self.bert_bfd_model = AutoModel.from_pretrained(BERT_BFD_MODEL)
        self.bert_bfd_tokenizer = AutoTokenizer.from_pretrained(BERT_BFD_MODEL, do_lower_case=False)
        self.bert_bfd_pipeline = pipeline(TASK, model = self.bert_bfd_model, tokenizer=self.bert_bfd_tokenizer, device=self.device)
        # XLNET
        self.xlnet_model = AutoModel.from_pretrained(XLNET_MODEL)
        self.xlnet_tokenizer = XLNetTokenizer.from_pretrained(XLNET_MODEL, do_lower_case=False)
        self.xlnet_pipeline = pipeline(TASK, model = self.xlnet_model, tokenizer=self.xlnet_tokenizer, device=self.device)

        # Private attributes with properties
        self._bert_embeddings: t.Dict[str, np.ndarray] = None
        self._albert_embeddings: t.Dict[str, np.ndarray] = None
        self._bert_bfd_embeddings: t.Dict[str, np.ndarray] = None
        self._xlnet_embeddings: t.Dict[str, np.ndarray] = None

        # Sequences
        self.sequences = [
            ' '.join(seq)
            for seq in self.data.sequences
        ]

    def compute_reduce_save_bert(self):
        """Computes the embeddings, reduce them with PCA"""

        logger.info(f'Computing BERT embeddings...')
        bert_embeddings_list = self.bert_pipeline(self.sequences)
        self._bert_embeddings = [
            np.mean(np.array(embedding[0][1:-1]), axis = 0)
            for embedding in bert_embeddings_list
        ]

        logger.info('Doing scaling and PCA for BERT embeddings...')
        # Scaling
        embeddings_df = pd.DataFrame(self._bert_embeddings)
        scaler = StandardScaler()
        scaled_embeddings  = scaler.fit_transform(embeddings_df)
        # PCA
        pca = PCA(n_components=N_COMPONENT_PCA)
        self._bert_embeddings = pca.fit_transform(scaled_embeddings)

        logger.info('Saving BERT embeddings...')
        pd.DataFrame(self._bert_embeddings).to_csv(project.get_new_embedding_file(BERT_EMBEDDING), index=False)

    def compute_reduce_save_albert(self):
        """Computes the embeddings, reduce them with PCA"""

        logger.info(f'Computing ALBERT embeddings...')
        albert_embeddings_list = self.albert_pipeline(self.sequences)
        self._albert_embeddings = [
            np.mean(np.array(embedding[0][1:-1]), axis = 0)
            for embedding in albert_embeddings_list
        ]

        logger.info('Doing scaling and PCA for ALBERT embeddings...')
        # Scaling
        embeddings_df = pd.DataFrame(self._albert_embeddings)
        scaler = StandardScaler()
        scaled_embeddings  = scaler.fit_transform(embeddings_df)
        # PCA
        pca = PCA(n_components=N_COMPONENT_PCA)
        self._albert_embeddings = pca.fit_transform(scaled_embeddings)

        logger.info('Saving ALBERT embeddings...')
        pd.DataFrame(self._albert_embeddings).to_csv(project.get_new_embedding_file(ALBERT_EMBEDDING), index=False)

    def compute_reduce_save_bert_bfd(self):
        """Computes the embeddings, reduce them with PCA"""

        logger.info(f'Computing BERT_BFD embeddings...')
        bert_bfd_embeddings_list = self.bert_bfd_pipeline(self.sequences)
        self._bert_bfd_embeddings = [
            np.mean(np.array(embedding[0][1:-1]), axis = 0)
            for embedding in bert_bfd_embeddings_list
        ]

        logger.info('Doing scaling and PCA for BERT_BFD embeddings...')
        # Scaling
        embeddings_df = pd.DataFrame(self._bert_bfd_embeddings)
        scaler = StandardScaler()
        scaled_embeddings  = scaler.fit_transform(embeddings_df)
        # PCA
        pca = PCA(n_components=N_COMPONENT_PCA)
        self._bert_bfd_embeddings = pca.fit_transform(scaled_embeddings)

        logger.info('Saving BERT_BFD embeddings...')
        pd.DataFrame(self._bert_bfd_embeddings).to_csv(project.get_new_embedding_file(BERT_BFD_EMBEDDING), index=False)

    def compute_reduce_save_xlnet(self):
        """Computes the embeddings, reduce them with PCA"""

        logger.info(f'Computing XLNET embeddings...')
        xlnet_embeddings_list = self.xlnet_pipeline(self.sequences)
        self._xlnet_embeddings = [
            np.mean(np.array(embedding[0][1:-1]), axis = 0)
            for embedding in xlnet_embeddings_list
        ]

        logger.info('Doing scaling and PCA for XLNET embeddings...')
        # Scaling
        embeddings_df = pd.DataFrame(self._xlnet_embeddings)
        scaler = StandardScaler()
        scaled_embeddings  = scaler.fit_transform(embeddings_df)
        # PCA
        pca = PCA(n_components=N_COMPONENT_PCA)
        self._xlnet_embeddings = pca.fit_transform(scaled_embeddings)

        logger.info('Saving XLNET embeddings...')
        pd.DataFrame(self._xlnet_embeddings).to_csv(project.get_new_embedding_file(XLNET_EMBEDDING), index=False)
    

def main():
    protbert = ProtBertClassifier()
    protbert.compute_reduce_save_bert()
    protbert.compute_reduce_save_albert()
    protbert.compute_reduce_save_bert_bfd()
    protbert.compute_reduce_save_xlnet()

if __name__ == '__main__':
    main()
