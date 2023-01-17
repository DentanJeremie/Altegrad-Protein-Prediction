import csv
import time
import numpy as np
import scipy.sparse as sp

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

from src.utils.constants import STRUCTRE_BASELINE_FEATURE_NAME as FEATURE_NAME
from src.utils.pathtools import project
from src.utils.logging import logger
from src.utils.structure_data import StructureData, structure_data, sparse_mx_to_torch_sparse_tensor

USE_PCA = False
EPOCH_VERBOSE = 5
EPOCHS = 300
BATCH_SIZE = 64
N_HIDDEN = 64
N_INPUT = 13 if USE_PCA else 86
DROPOUT = 0.2
LEARNING_RATE = 0.001
N_CLASS = 18

class GNN(nn.Module):
    """
    Simple message passing model that consists of 2 message passing layers
    and the sum aggregation function
    """
    def __init__(self, input_dim, hidden_dim, dropout, n_class):
        super(GNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, n_class)
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_in, adj, idx):
        # first message passing layer
        x = self.fc1(x_in)
        x = self.relu(torch.mm(adj, x))
        x = self.dropout(x)

        # second message passing layer
        x = self.fc2(x)
        x = self.relu(torch.mm(adj, x))
        
        # sum aggregator
        idx = idx.unsqueeze(1).repeat(1, x.size(1))
        out = torch.zeros(torch.max(idx)+1, x.size(1)).to(x_in.device)
        out = out.scatter_add_(0, idx, x)
        
        # batch normalization layer
        out = self.bn(out)

        # mlp to produce output
        out = self.relu(self.fc3(out))
        out = self.dropout(out)
        out = self.fc4(out)

        return F.log_softmax(out, dim=1)

class StructureBaseline():

    def __init__(self, data: StructureData = structure_data):
        # Data
        self.data = data
        self._adj_train = None
        self._features_train = None
        self._y_train = None
        self._proteins_train = None
        self._adj_test = None
        self._features_test = None
        self._proteins_test = None
        # Model
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.model = GNN(N_INPUT, N_HIDDEN, DROPOUT, N_CLASS).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
        self.loss_function = nn.CrossEntropyLoss()
        # Training
        self._training_done = False

# ------------------ PROPERTIES ------------------

    @property
    def adj_train(self):
        if self._adj_train is None:
            self.split_train_test()
        return self._adj_train

    @property
    def features_train(self):
        if self._features_train is None:
            self.split_train_test()
        return self._features_train

    @property
    def y_train(self):
        if self._y_train is None:
            self.split_train_test()
        return self._y_train

    @property
    def proteins_train(self):
        if self._proteins_train is None:
            self.split_train_test()
        return self._proteins_train

    @property
    def adj_test(self):
        if self._adj_test is None:
            self.split_train_test()
        return self._adj_test

    @property
    def features_test(self):
        if self._features_test is None:
            self.split_train_test()
        return self._features_test

    @property
    def proteins_test(self):
        if self._proteins_test is None:
            self.split_train_test()
        return self._proteins_test

# ------------------ SPLIT TRAIN TEST ------------------

    def split_train_test(self):
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

        # PCA ?
        if USE_PCA:
            node_features = self.data.reduced_node_features
        else:
            node_features = self.data.node_features

        self._adj_train = list()
        self._features_train = list()
        self._y_train = list()
        self._proteins_train = list()
        self._adj_test = list()
        self._features_test = list()
        self._proteins_test = list()

        with project.graph_labels.open('r') as f:
            for i,line in enumerate(f):
                t = line.split(',')
                if len(t[1][:-1]) == 0:
                    self._proteins_test.append(t[0])
                    self._adj_test.append(self.data.adjacency_matrixes[i])
                    self._features_test.append(node_features[i])
                else:
                    self._proteins_train.append(t[0])
                    self._adj_train.append(self.data.adjacency_matrixes[i])
                    self._features_train.append(node_features[i])
                    self._y_train.append(int(t[1][:-1]))

# ------------------ SPLIT TRAIN TEST ------------------

    def train(self):
        """Trains the model.
        """
        logger.info('Starting the training of the model...')
        N_train = len(self.adj_train)

        for epoch in range(EPOCHS):
            t = time.time()
            self.model.train()
            train_loss = 0
            correct = 0
            count = 0
            # Iterate over the batches
            for i in range(0, N_train, BATCH_SIZE):
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
                output = self.model(features_batch, adj_batch, idx_batch)
                loss = self.loss_function(output, y_batch)
                train_loss += loss.item() * output.size(0)
                count += output.size(0)
                preds = output.max(1)[1].type_as(y_batch)
                correct += torch.sum(preds.eq(y_batch).double())
                loss.backward()
                self.optimizer.step()
            
            if epoch % EPOCH_VERBOSE == 0:
                logger.info(
                    f"Epoch: {epoch+1:03d}  "
                    f"loss_train: {train_loss / count:.4f}  "
                    f"acc_train: {correct / count:.4f}  "
                    f"time: {time.time() - t:.4f}s  "
                )

        logger.info('Training: done !')
        self._training_done = True

    def predict(self):
        """Makes the prediction and saves it.
        """
        if not self._training_done:
            self.train()

        logger.info('Starting the prediction...')
        N_test = len(self.adj_test)

        self.model.eval()
        y_pred_proba = list()
        # Iterate over the batches
        for i in range(0, N_test, BATCH_SIZE):
            adj_batch = list()
            idx_batch = list()
            features_batch = list()
            y_batch = list()
            
            # Create tensors
            for j in range(i, min(N_test, i+BATCH_SIZE)):
                n = self.adj_test[j].shape[0]
                adj_batch.append(self.adj_test[j]+sp.identity(n))
                features_batch.append(self.features_test[j])
                idx_batch.extend([j-i]*n)
                
            adj_batch = sp.block_diag(adj_batch)
            features_batch = np.vstack(features_batch)

            adj_batch = sparse_mx_to_torch_sparse_tensor(adj_batch).to(self.device)
            features_batch = torch.FloatTensor(features_batch).to(self.device)
            idx_batch = torch.LongTensor(idx_batch).to(self.device)

            output = self.model(features_batch, adj_batch, idx_batch)
            y_pred_proba.append(output)
            
        y_pred_proba = torch.cat(y_pred_proba, dim=0)
        y_pred_proba = torch.exp(y_pred_proba)
        y_pred_proba = y_pred_proba.detach().cpu().numpy()

        # Write predictions to a file
        output_path = project.get_new_feature_file(FEATURE_NAME)
        with output_path.open('w') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            lst = list()
            for i in range(18):
                lst.append('class'+str(i))
            lst.insert(0, "name")
            writer.writerow(lst)
            for i, protein in enumerate(self.proteins_test):
                lst = y_pred_proba[i,:].tolist()
                lst.insert(0, protein)
                writer.writerow(lst)

        logger.info(f'Prediction stored at {project.as_relative(output_path)}')

def main():
    structure_baseline = StructureBaseline()
    structure_baseline.predict()

if __name__ == '__main__':
    main()