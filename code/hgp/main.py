import argparse
import glob
import os
import time
import csv

import numpy as np

import torch
import torch.nn.functional as F
from models import Model
from torch.utils.data import random_split
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import TUDataset

from utils import load_data, normalize_adjacency

parser = argparse.ArgumentParser()

# Hyperparameters
epochs = 100
batch_size = 64
n_hidden = 64
n_input = 86
dropout = 0.2
learning_rate = 0.001
n_class = 18

#Initialize device
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

parser.add_argument('--seed', type=int, default=777, help='random seed')
parser.add_argument('--batch_size', type=int, default=batch_size, help='batch size')
parser.add_argument('--lr', type=float, default=learning_rate, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=0.001, help='weight decay')
parser.add_argument('--nhid', type=int, default=n_hidden, help='hidden size')
parser.add_argument('--sample_neighbor', type=bool, default=True, help='whether sample neighbors')
parser.add_argument('--sparse_attention', type=bool, default=True, help='whether use sparse attention')
parser.add_argument('--structure_learning', type=bool, default=True, help='whether perform structure learning')
parser.add_argument('--pooling_ratio', type=float, default=0.5, help='pooling ratio')
parser.add_argument('--dropout_ratio', type=float, default=dropout, help='dropout ratio')
parser.add_argument('--lamb', type=float, default=1.0, help='trade-off parameter')
parser.add_argument('--dataset', type=str, default='PROTEINS', help='DD/PROTEINS/NCI1/NCI109/Mutagenicity/ENZYMES')
parser.add_argument('--device', type=str, default=device, help='specify cuda devices')
parser.add_argument('--epochs', type=int, default=epochs, help='maximum number of epochs')
parser.add_argument('--patience', type=int, default=100, help='patience for early stopping')

args = parser.parse_args()
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)

path = '/Users/meryem/Documents/GitHub/Altegrad-Protein-Prediction/data/'
adj, features, edge_features, edge_index = load_data()

adj = [normalize_adjacency(A) for A in adj]

# Split data into training and test sets
adj_train = list()
features_train = list()
y_train = list()
edge_index_train = list()
edge_features_train = list()

adj_test = list()
features_test = list()
proteins_test = list()
edge_index_test = list()
edge_features_test = list()

with open(path+'graph_labels.txt', 'r') as f:
    for i,line in enumerate(f):
        t = line.split(',')
        if len(t[1][:-1]) == 0:
            proteins_test.append(t[0])
            adj_test.append(torch.from_numpy(adj[i].toarray()))
            features_test.append(torch.from_numpy(features[i]).float())
            edge_index_test.append(edge_index[i])
            edge_features_test.append(torch.from_numpy(edge_features[i]))
        else:
            adj_train.append(torch.from_numpy(adj[i].toarray()))
            features_train.append(torch.from_numpy(features[i]).float())
            y_train.append(torch.from_numpy(np.array(int(t[1][:-1]))))
            edge_index_train.append(edge_index[i])
            edge_features_train.append(torch.from_numpy(edge_features[i]))


dataset = [Data(x=features_train[i], edge_index = edge_index_train[i], edge_attr = edge_features_train[i], y = y_train[i]) for i in range(len(y_train))]
test_set = [Data(x=features_test[i], edge_index = edge_index_test[i], edge_attr = edge_features_test[i]) for i in range(len(proteins_test))]

args.num_classes = n_class
args.num_features = n_input

num_training = int(len(dataset) * 0.9)
num_val = len(dataset) - num_training
training_set, validation_set = random_split(dataset, [num_training, num_val])

train_loader = DataLoader(training_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(validation_set, batch_size=args.batch_size, shuffle=False)
test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

model = Model(args).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

def train():
    min_loss = 1e10
    patience_cnt = 0
    val_loss_values = []
    best_epoch = 0

    t = time.time()
    model.train()
    for epoch in range(args.epochs):
        loss_train = 0.0
        correct = 0
        for i, data in enumerate(train_loader):
            optimizer.zero_grad()
            data = data.to(device)
            out = model(data)
            loss = F.nll_loss(out, data.y)
            loss.backward()
            optimizer.step()
            loss_train += loss.item()
            pred = out.max(dim=1)[1]
            correct += pred.eq(data.y).sum().item()
        acc_train = correct / len(train_loader.dataset)
        acc_val, loss_val = compute_test(val_loader)
        print('Epoch: {:04d}'.format(epoch + 1), 'loss_train: {:.6f}'.format(loss_train),
              'acc_train: {:.6f}'.format(acc_train), 'loss_val: {:.6f}'.format(loss_val),
              'acc_val: {:.6f}'.format(acc_val), 'time: {:.6f}s'.format(time.time() - t))

        val_loss_values.append(loss_val)
        torch.save(model.state_dict(), '{}.pth'.format(epoch))
        if val_loss_values[-1] < min_loss:
            min_loss = val_loss_values[-1]
            best_epoch = epoch
            patience_cnt = 0
        else:
            patience_cnt += 1

        if patience_cnt == args.patience:
            break

        files = glob.glob('*.pth')
        for f in files:
            epoch_nb = int(f.split('.')[0])
            if epoch_nb < best_epoch:
                os.remove(f)

    files = glob.glob('*.pth')
    for f in files:
        epoch_nb = int(f.split('.')[0])
        if epoch_nb > best_epoch:
            os.remove(f)
    print('Optimization Finished! Total time elapsed: {:.6f}'.format(time.time() - t))

    return best_epoch


def compute_test(loader):
    model.eval()
    correct = 0.0
    loss_test = 0.0
    for data in loader:
        data = data.to(args.device)
        out = model(data)
        pred = out.max(dim=1)[1]
        correct += pred.eq(data.y).sum().item()
        loss_test += F.nll_loss(out, data.y).item()
    return correct / len(loader.dataset), loss_test

def compute_test_final(loader):
    model.eval()
    y_pred_proba = list()
    for data in loader:
        data = data.to(args.device)
        out = model(data)
        y_pred_proba.append(out)
    y_pred_proba = torch.cat(y_pred_proba, dim=0)
    y_pred_proba = torch.exp(y_pred_proba)
    y_pred_proba = y_pred_proba.detach().cpu().numpy()
    return y_pred_proba


if __name__ == '__main__':
    # Model training
    best_model = train()
    # Restore best model for test set
    model.load_state_dict(torch.load('{}.pth'.format(best_model)))
    y_pred_proba = compute_test_final(test_loader)
    # Write predictions to a file
    with open('sample_submissionHGP11.csv', 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        lst = list()
        for i in range(18):
            lst.append('class'+str(i))
        lst.insert(0, "name")
        writer.writerow(lst)
        for i, protein in enumerate(proteins_test):
            lst = y_pred_proba[i,:].tolist()
            lst.insert(0, protein)
            writer.writerow(lst)