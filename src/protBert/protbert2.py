"""
This scripts implements the computation of embeddings from the sequences.
It uses a pretrained model, and is inspired from a code taken in https://github.com/agemagician/ProtTrans
"""

import time
import typing as t

import numpy as np
import torch
from transformers import T5EncoderModel, T5Tokenizer

from src.utils.pathtools import project
from src.utils.logging import logger
from src.utils.sequence_data import SequenceData, sequence_data
from src.utils.train_validation_test import SetsManager, sets_manager

MODEL_NAME = 'Rostlab/prot_t5_xl_half_uniref50-enc'
MAX_RESIDUES = 4000
MAX_SEQ_LEN = 1000
MAX_BATCH = 100


class ProtBertClassifier():

    def __init__(
        self,
        data: SequenceData = sequence_data,
        sets: SetsManager = sets_manager,
    ):
        # Data
        self.data = data
        self.sets = sets
        
        # Building model
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model = T5EncoderModel.from_pretrained(MODEL_NAME)
        self.tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME, do_lower_case=False)
        self.model = self.model.to(self.device) # move self.model to GPU
        self.model = self.model.eval() # set self.model to evaluation model

        # Private attributes with properties
        self._sequences_dict: t.Dict[int, str] = None
        self._embeddings: t.Dict[str, np.ndarray] = None

    @property
    def sequences_dict(self):
        if self._sequences_dict is None:
            self.build_sequence_dict()
        return self._sequences_dict

    @property
    def embeddings(self):
        if self._embeddings is None:
            self.compute_embeddings()
        return self._embeddings

    def build_sequence_dict(self):
        """Builds the sequence dict.
        It is a dict {protein_index -> protein_sequence}
        """
        self._sequences_dict = {index: item for index, item in enumerate(self.data.sequences[:10])}
    
    def compute_embeddings(self):
        """Computes the embeddings of the proteins.
        After execution, self.results is a dict{protein_index -> protein_embedding}
        The embeddings are np.ndarray of size 1024.
        """
        logger.info(f'Computing embedding for {len(self.sequences)} protein sequences...')
        # sort sequences according to length (reduces unnecessary padding --> speeds up embedding)
        seq_dict   = sorted(
            self.sequences.items(),
            key=lambda kv: len(self.sequences[kv[0]] ),
            reverse=True,
        )

        # Batch computation
        self.embeddings: t.Dict[int, np.ndarray] = dict() 
        start = time.time()
        batch = list()
        for seq_idx, (pdb_id, seq) in enumerate(seq_dict, 1):
            seq = seq
            seq_len = len(seq)
            seq = ' '.join(list(seq))
            batch.append((pdb_id,seq,seq_len))

            # count residues in current batch and add the last sequence length to
            # avoid that batches with (n_res_batch > max_residues) get processed 
            n_res_batch = sum([s_len for  _, _, s_len in batch ]) + seq_len 
            if len(batch) >= MAX_BATCH or n_res_batch>=MAX_RESIDUES or seq_idx==len(seq_dict) or seq_len>MAX_SEQ_LEN:
                pdb_ids, seqs, seq_lens = zip(*batch)
                batch = list()

                # add_special_tokens adds extra token at the end of each sequence
                token_encoding = self.tokenizer.batch_encode_plus(seqs, add_special_tokens=True, padding="longest")
                input_ids      = torch.tensor(token_encoding['input_ids']).to(self.device)
                attention_mask = torch.tensor(token_encoding['attention_mask']).to(self.device)
                
                try:
                    with torch.no_grad():
                        embedding_repr = self.model(input_ids, attention_mask=attention_mask)
                except RuntimeError:
                    logger.error("RuntimeError during embedding for {} (L={})".format(pdb_id, seq_len))
                    continue

                for batch_idx, identifier in enumerate(pdb_ids): # for each protein in the current mini-batch
                    s_len = seq_lens[batch_idx]
                    # slice off padding --> batch-size x seq_len x embedding_dim  
                    emb = embedding_repr.last_hidden_state[batch_idx,:s_len]
                    # apply average-pooling to derive per-protein embeddings (1024-d)
                    protein_emb = emb.mean(dim=0)

                    # Saving the result
                    self.embeddings[identifier] = protein_emb.detach().cpu().numpy().squeeze()

        # Timing
        logger.info('Computation of the embedding of the proteins: done!')
        passed_time=time.time()-start
        avg_time = passed_time/len(self.embeddings)
        logger.info('Total number of per-protein embeddings: {}'.format(len(self.embeddings)))
        logger.info("Time for generating embeddings: {:.1f}[m] ({:.3f}[s/protein])".format(
            passed_time/60, avg_time ))

protein = ProtBertClassifier()