import csv
import time
import typing as t

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

from src.utils.constants import *
from src.utils.pathtools import project
from src.utils.logging import logger
from src.utils.structure_data import StructureData, structure_data, sparse_mx_to_torch_sparse_tensor
from src.utils.train_validation_test import SetsManager, sets_manager
from src.embeddings.graph_models import GNN, DGCNN

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_add_pool as gap


"""
In this module, we extend the StructureBaseline class in the gnn.py module in order to use other architectures.
"""
USE_PCA = False
EPOCHS = 200
BATCH_SIZE = 64
N_HIDDEN = 96
N_INPUT = 13 if USE_PCA else 86
DROPOUT = 0.2
LEARNING_RATE = 0.001
N_CLASS = 18
PATIENCE = 10
MIN_EPOCH = 50
N_COMPONENT_PCA = 32


class Structure():

    def __init__(self, model, device, data: StructureData = structure_data, sets: SetsManager = sets_manager):
        # Data
        self.data = data
        self.sets = sets

        self.model = model
        self.device = device
        self.pyg_format = self.model.pyg_format
        self.optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
        self.loss_function = nn.CrossEntropyLoss()

        # Best performance
        self.epoch_for_best_evaluation = -1
        self.best_evaluation_loss = -1
        self.weights_for_best_evaluation = None

# ------------------ SPLIT TRAIN TEST ------------------

    
    
    def split_train_validation(self):
        """Splits the dataset into train and test.
        The following attributes are built:
        * `self.adj_train`
        * `self.features_train`
        * `self.y_train`
        * `self.proteins_train`
        * `self.adj_test`
        * `self.features_test`
        * `self.proteins_test`
        """

        logger.info('Getting data and separating between train and validation...')

        

        # We only need to separate the data for train / validation
        # Indeed, at the end we compute the embedding for everybody,
        # not only for the test set

        # Train
        self.adj_train: t.List[sp.csr_matrix] = list()
        self.features_train: t.List[np.ndarray] = list()
        self.y_train: t.List[int] = list()
        self.proteins_train: t.List[str] = list()
        self.edge_features_train: t.List[np.ndarray] = list()
        self.edge_index_train : t.List[torch.Tensor] = list()
        # Validation
        self.adj_validation: t.List[sp.csr_matrix] = list()
        self.features_validation: t.List[np.ndarray] = list()
        self.edge_features_validation: t.List[np.ndarray] = list()
        self.y_validation: t.List[int] = list()
        self.proteins_validation: t.List[str] = list()
        self.edge_index_validation : t.List[torch.Tensor] = list()
        # Full
        self.adj_full: t.List[sp.csr_matrix] = list()
        self.features_full: t.List[np.ndarray] = list()
        self.edge_features_full: t.List[np.ndarray] = list()
        self.edge_index_full : t.List[torch.Tensor] = list()
        self.proteins_full: t.List[str] = list()

        if self.pyg_format:
            self.load_data_hgp()
        else:
            self.load_data()

    
    def load_data(self):
        # PCA ?
        if USE_PCA:
            node_features = self.data.reduced_node_features
        else:
            node_features = self.data.node_features

        for index in range(len(node_features)):
            # Train
            if self.sets.is_train(index):
                self.adj_train.append(self.data.adjacency_matrixes[index])
                self.features_train.append(node_features[index])
                self.y_train.append(self.sets.get_label(index))
                self.proteins_train.append(self.sets.index_to_protein(index))

            # Validation
            if self.sets.is_validation(index):
                self.adj_validation.append(self.data.adjacency_matrixes[index])
                self.features_validation.append(node_features[index])
                self.y_validation.append(self.sets.get_label(index))
                self.proteins_validation.append(self.sets.index_to_protein(index))

            # Full
            self.adj_full.append(self.data.adjacency_matrixes[index])
            self.features_full.append(node_features[index])
            self.proteins_full.append(self.sets.index_to_protein(index))
            

    def load_data_hgp(self):
        # PCA ?
        if USE_PCA:
            node_features = self.data.reduced_node_features_hgp
        else:
            node_features = self.data.node_features_hgp
        for index in range(len(node_features)):
            # Train
            if self.sets.is_train(index):
                #self.adj_train.append(self.data.adjacency_matrixes_hgp[index])
                self.features_train.append(torch.from_numpy(node_features[index]).float())
                self.edge_features_train.append(torch.from_numpy(self.data.edge_features_hgp[index]).float())
                self.edge_index_train.append(self.data.edge_index_hgp[index])
                self.y_train.append(self.sets.get_label(index))
                self.proteins_train.append(self.sets.index_to_protein(index))

            # Validation
            if self.sets.is_validation(index):
                #self.adj_validation.append(self.data.adjacency_matrixes_hgp[index])
                self.features_validation.append(torch.from_numpy(node_features[index]).float())
                self.edge_features_validation.append(torch.from_numpy(self.data.edge_features_hgp[index]).float())
                self.edge_index_validation.append(self.data.edge_index_hgp[index])
                self.y_validation.append(self.sets.get_label(index))
                self.proteins_validation.append(self.sets.index_to_protein(index))

            # Full
            #self.adj_full.append(self.data.adjacency_matrixes_hgp[index])
            self.features_full.append(torch.from_numpy(node_features[index]).float())
            self.edge_features_full.append(torch.from_numpy(self.data.edge_features_hgp[index]).float())
            self.edge_index_full.append(self.data.edge_index_hgp[index])
            self.proteins_full.append(self.sets.index_to_protein(index))
        
        self.training_data = [Data(x = self.features_train[i], edge_index = self.edge_index_train[i], 
            edge_attr = self.edge_features_train[i], y = self.y_train[i]) for i in range(len(self.y_train))]
        self.val_data = [Data(x=self.features_validation[i], edge_index = self.edge_index_validation[i], 
            edge_attr = self.edge_features_validation[i], y = self.y_validation[i]) for i in range(len(self.y_validation))]
        self.train_loader = DataLoader(self.training_data, batch_size=BATCH_SIZE, shuffle=True)
        self.val_loader = DataLoader(self.val_data, batch_size=BATCH_SIZE, shuffle=False)


# ------------------ TRAIN ------------------

    def train(self):
        if self.pyg_format:
            self.train_pyg()
        else:
            self.train_standard()
    
    def train_standard(self):
        """Trains the model.
        """
        logger.info('Starting the training of the model...')
        N_train = len(self.adj_train)

        validation_losses = list()
        validation_corrects = list()

        for epoch in range(EPOCHS):
            t = time.time()
            train_loss = 0
            correct = 0
            count = 0

            # Iterate over the batches
            for i in range(0, N_train, BATCH_SIZE):
                self.model.train()

                adj_batch = list()
                features_batch = list()
                idx_batch = list()
                y_batch = list()
                
                # Create tensors
                for j in range(i, min(N_train, i+BATCH_SIZE)):
                    n = self.adj_train[j].shape[0]
                    adj_batch.append(self.adj_train[j]+sp.identity(n))
                    features_batch.append(self.features_train[j])
                    idx_batch.extend([j-i]*n)
                    y_batch.append(self.y_train[j])
                    
                adj_batch = sp.block_diag(adj_batch)
                features_batch = np.vstack(features_batch)

                adj_batch = sparse_mx_to_torch_sparse_tensor(adj_batch).to(self.device)
                features_batch = torch.FloatTensor(features_batch).to(self.device)
                idx_batch = torch.LongTensor(idx_batch).to(self.device)
                y_batch = torch.LongTensor(y_batch).to(self.device)
                
                self.optimizer.zero_grad()
                output, _, _ = self.model(features_batch, adj_batch, idx_batch)
                loss = self.loss_function(output, y_batch)
                train_loss += loss.item() * output.size(0)
                count += output.size(0)
                preds = output.max(1)[1].type_as(y_batch)
                correct += torch.sum(preds.eq(y_batch).double())
                loss.backward()
                self.optimizer.step()

            valid_loss, valid_correct = self.eval()
            validation_losses.append(valid_loss)
            validation_corrects.append(valid_correct)

            if valid_loss < self.best_evaluation_loss or self.best_evaluation_loss < 0:
                self.best_evaluation_loss = valid_loss
                self.weights_for_best_evaluation = self.model.state_dict()
                self.epoch_for_best_evaluation = epoch
            
            logger.info(
                f"Epoch: {epoch+1:03d}  "
                f"loss_train: {train_loss / count:.4f}  "
                f"acc_train: {correct / count:.4f}  "
                f"loss_valid: {valid_loss:.4f}  "
                f"acc_valid: {valid_correct:.4f}  "
                f"time: {time.time() - t:.4f}s  "
            )

            if len(validation_losses) > MIN_EPOCH and min(validation_losses[-4:-1]) > validation_losses[-5]:
                logger.info('Overfitting detected (patience = 5). Stopping training.')
                break

        logger.info('Training: done !')
        self._training_done = True

    def train_pyg(self):
        """Trains the model in pyg format.
        """
        logger.info('Starting the training of the model...')
        N_train = len(self.adj_train)
        validation_losses = list()
        validation_corrects = list()

        for epoch in range(EPOCHS):
            t = time.time()
            train_loss = 0
            correct = 0
            count = 0
            # Iterate over the batches
            for data_batch in self.train_loader:
                self.model.train()
                data_batch = data_batch.to(self.device)
                self.optimizer.zero_grad()
                output,_,_ = self.model(data_batch)
                loss = self.loss_function(output, data_batch.y)
                train_loss += loss.item() * output.size(0)
                count += output.size(0)
                preds = output.max(1)[1].type_as(data_batch.y)
                correct += torch.sum(preds.eq(data_batch.y).double())
                loss.backward()
                self.optimizer.step()

            valid_loss, valid_correct = self.eval_pyg()
            validation_losses.append(valid_loss)
            validation_corrects.append(valid_correct)
            if valid_loss < self.best_evaluation_loss or self.best_evaluation_loss < 0:
                self.best_evaluation_loss = valid_loss
                self.weights_for_best_evaluation = self.model.state_dict()
                self.epoch_for_best_evaluation = epoch

            logger.info(
                f"Epoch: {epoch+1:03d}  "
                f"loss_train: {train_loss / count:.4f}  "
                f"acc_train: {correct / count:.4f}  "
                f"loss_valid: {valid_loss:.4f}  "
                f"acc_valid: {valid_correct:.4f}  "
                f"time: {time.time() - t:.4f}s  "
            )

            if len(validation_losses) > MIN_EPOCH and min(validation_losses[-4:-1]) > validation_losses[-5]:
                logger.info('Overfitting detected (patience = 5). Stopping training.')
                break

        logger.info('Training: done !')
        self._training_done = True
    
    def eval(self) -> t.Tuple[float, float]:
        """Returns the log loss and the proportion of correct predictions on the validation set.

        :returns: The tuple(log_loss, prop_correct_preds)
        """
        self.model.eval()

        N_validation = len(self.adj_validation)  
        total_loss = 0
        total_correct = 0
        count = 0
        t = time.time()

        # Iterate over the batches
        for i in range(0, N_validation, BATCH_SIZE):
            adj_batch = list()
            features_batch = list()
            idx_batch = list()
            y_batch = list()
            
            # Create tensors
            for j in range(i, min(N_validation, i+BATCH_SIZE)):
                n = self.adj_validation[j].shape[0]
                adj_batch.append(self.adj_validation[j]+sp.identity(n))
                features_batch.append(self.features_validation[j])
                idx_batch.extend([j-i]*n)
                y_batch.append(self.y_validation[j])
                
            adj_batch = sp.block_diag(adj_batch)
            features_batch = np.vstack(features_batch)

            adj_batch = sparse_mx_to_torch_sparse_tensor(adj_batch).to(self.device)
            features_batch = torch.FloatTensor(features_batch).to(self.device)
            idx_batch = torch.LongTensor(idx_batch).to(self.device)
            y_batch = torch.LongTensor(y_batch).to(self.device)
            
            output, _, _ = self.model(features_batch, adj_batch, idx_batch)
            loss = self.loss_function(output, y_batch)
            total_loss += loss.item() * output.size(0)
            preds = output.max(1)[1].type_as(y_batch)
            total_correct += torch.sum(preds.eq(y_batch).double())
            count += output.size(0)
        
        return (total_loss / count, total_correct / count)
    
    def eval_pyg(self):
        """Returns the log loss and the proportion of correct predictions on the validation set.

        :returns: The tuple(log_loss, prop_correct_preds)
        """
        self.model.eval()

        N_validation = len(self.adj_validation)  
        total_loss = 0
        total_correct = 0
        count = 0
        t = time.time()
        # Iterate over the batches
        for data_batch in self.val_loader:
            data_batch = data_batch.to(self.device)
            output, _, _ = self.model(data_batch)
            loss = self.loss_function(output, data_batch.y)
            total_loss += loss.item() * output.size(0)
            preds = output.max(1)[1].type_as(data_batch.y)
            total_correct += torch.sum(preds.eq(data_batch.y).double())
            count += output.size(0)
        
        return (total_loss / count, total_correct / count)

        
# ------------------ PREDICT ------------------
    def predict(self):
        if self.pyg_format:
            self.predict_pyg()
        else:
            self.predict_standard()


    def predict_standard(self):
        """Makes the prediction and saves it.
        """

        logger.info('Computing the embeddings of all graphs...')
        N_full = len(self.adj_full)

        logger.info(f'Loading best model from epoch {self.epoch_for_best_evaluation +1}')
        self.model.load_state_dict(self.weights_for_best_evaluation)
        self.model.eval()

        embeddings_last_list = list()
        embeddings_previous_list = list()
        # Iterate over the batches
        for i in range(0, N_full, BATCH_SIZE):
            adj_batch = list()
            idx_batch = list()
            features_batch = list()
            
            # Create tensors
            for j in range(i, min(N_full, i+BATCH_SIZE)):
                n = self.adj_full[j].shape[0]
                adj_batch.append(self.adj_full[j]+sp.identity(n))
                features_batch.append(self.features_full[j])
                idx_batch.extend([j-i]*n)
                
            adj_batch = sp.block_diag(adj_batch)
            features_batch = np.vstack(features_batch)

            adj_batch = sparse_mx_to_torch_sparse_tensor(adj_batch).to(self.device)
            features_batch = torch.FloatTensor(features_batch).to(self.device)
            idx_batch = torch.LongTensor(idx_batch).to(self.device)

            _, embedding_last, embedding_previous = self.model(features_batch, adj_batch, idx_batch)
            embeddings_last_list.append(embedding_last)
            embeddings_previous_list.append(embedding_previous)

        for embeddings_list, name in zip(
            (embeddings_last_list, embeddings_previous_list),
            (GNN_EMBEDDING_LAST, GNN_EMBEDDING_PREVIOUS)
        ):
            embeddings = torch.cat(embeddings_list, dim = 0)
            embeddings = embeddings.cpu().detach().numpy()

            if embeddings.shape[1] > 32:
                logger.info(f'Doing PCA on the {name} embeddings (dim {embeddings.shape[1]} -> {N_COMPONENT_PCA})...')
                # Scaling
                embeddings_df = pd.DataFrame(embeddings)
                scaler = StandardScaler()
                scaled_embeddings  = scaler.fit_transform(embeddings_df)
                # PCA
                pca = PCA(n_components=N_COMPONENT_PCA)
                reduced_embeddings = pca.fit_transform(scaled_embeddings)

            else:
                logger.info(f'No PCA: embeddings dimension is {embeddings.shape[1]} <= {N_COMPONENT_PCA}')
                reduced_embeddings = embeddings

            logger.info(f'Saving {name} embeddings...')
            output_path = project.get_new_embedding_file(name)
            pd.DataFrame(reduced_embeddings).to_csv(output_path, index=False)

    def predict_pyg(self):
        """Makes the prediction and saves it.
        """
        logger.info('Computing the embeddings of all graphs...')
        N_full = len(self.features_full)

        logger.info(f'Loading best model from epoch {self.epoch_for_best_evaluation +1}')
        self.model.load_state_dict(self.weights_for_best_evaluation)
        self.model.eval()

        embeddings_last_list = list()
        embeddings_previous_list = list()
        # Creating dataloader
        full_data = [Data(x = self.features_full[i], edge_index = self.edge_index_full[i], 
            edge_attr = self.edge_features_full[i]) for i in range(N_full)]
        full_loader = DataLoader(full_data, batch_size=BATCH_SIZE, shuffle=True)

        # Iterate over the batches
        for data_batch in full_loader:
            data_batch = data_batch.to(self.device)
            _, embedding_last, embedding_previous = self.model(data_batch)
            embeddings_last_list.append(embedding_last)
            embeddings_previous_list.append(embedding_previous)

        for embeddings_list, name in zip(
            (embeddings_last_list, embeddings_previous_list),
            (GNN_EMBEDDING_LAST, GNN_EMBEDDING_PREVIOUS)
        ):
            embeddings = torch.cat(embeddings_list, dim = 0)
            embeddings = embeddings.cpu().detach().numpy()

            if embeddings.shape[1] > 32:
                logger.info(f'Doing PCA on the {name} embeddings (dim {embeddings.shape[1]} -> {N_COMPONENT_PCA})...')
                # Scaling
                embeddings_df = pd.DataFrame(embeddings)
                scaler = StandardScaler()
                scaled_embeddings  = scaler.fit_transform(embeddings_df)
                # PCA
                pca = PCA(n_components=N_COMPONENT_PCA)
                reduced_embeddings = pca.fit_transform(scaled_embeddings)

            else:
                logger.info(f'No PCA: embeddings dimension is {embeddings.shape[1]} <= {N_COMPONENT_PCA}')
                reduced_embeddings = embeddings

            logger.info(f'Saving {name} embeddings...')
            output_path = project.get_new_embedding_file(name)
            pd.DataFrame(reduced_embeddings).to_csv(output_path, index=False)



def main():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    #model = GNN(N_INPUT, N_HIDDEN, DROPOUT, N_CLASS).to(device)
    model = DGCNN(N_INPUT, N_CLASS).to(device)
    data = StructureData()
    structure= Structure(model, device, data)
    structure.split_train_validation()
    structure.train()
    structure.predict()

if __name__ == '__main__':
    main()