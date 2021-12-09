"""
    File name: loader_representations.py
    Author: Pedro Ramoneda
    Python Version: 3.7
"""
import os
import json
import sys

import numpy as np
import pandas as pd
import sklearn
import torch
from matplotlib import pyplot as plt
from scipy.stats import spearmanr, kendalltau
from sequentia import Standardize, KNNClassifier, GMMHMM, HMMClassifier
from sklearn.metrics import confusion_matrix, balanced_accuracy_score
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.preprocessing import minmax_scale
from tqdm import tqdm
from sklearn import preprocessing
import torch.nn.functional as F

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence as packer, pad_packed_sequence as padder


# ----------------------------------------------------------------------------------------------------------------------
import utils
from loader_representations import load_rep, load_rep_info
from utils import load_json, save_json


def get_split(random_state, X, paths):

    split = load_json("mikrokosmos/splits.json")[str(random_state)]
    X_train, X_test, y_train, y_test, ids_train, ids_test = [], [], [], [], [], []
    # train
    for y, idx in zip(split['y_train'], split['ids_train']):
        index = np.where(paths == idx)[0][0]
        X_train.append(X[index])
        y_train.append(y)
        ids_train.append(int(os.path.basename(idx)[:-4]))

    # test
    for y, idx in zip(split['y_test'], split['ids_test']):
        index = np.where(paths == idx)[0][0]
        X_test.append(X[index])
        y_test.append(y)
        ids_test.append(int(os.path.basename(idx)[:-4]))
    return np.array(X_train), np.array(X_test), np.array(y_train), np.array(y_test), np.array(ids_train), np.array(ids_test)


def collate_fn(batch):
    """Collects together univariate or multivariate sequences into a single batch, arranged in descending order of length.
    Also returns the corresponding lengths and labels as :class:`torch:torch.LongTensor` objects.
    Parameters
    ----------
    batch: list of tuple(torch.FloatTensor, int)
        Collection of :math:`B` sequence-label pairs, where the :math:`n^\\text{th}` sequence is of shape :math:`(T_n \\times D)` or :math:`(T_n,)` and the label is an integer.
    Returns
    -------
    padded_sequences: :class:`torch:torch.Tensor` (float)
        A tensor of size :math:`B \\times T_\\text{max} \\times D` containing all of the sequences in descending length order, padded to the length of the longest sequence in the batch.
    lengths: :class:`torch:torch.Tensor` (long/int)
        A tensor of the :math:`B` sequence lengths in descending order.
    labels: :class:`torch:torch.Tensor` (long/int)
        A tensor of the :math:`B` sequence labels in descending length order.
    """
    batch_size = len(batch)

    # Sort the (sequence, label) pairs in descending order of duration
    batch.sort(key=(lambda x: len(x[0])), reverse=True)
    # Shape: list(tuple(tensor(TxD), int)) or list(tuple(tensor(T), int))

    # Create list of sequences, and tensors for lengths and labels
    sequences, lengths, labels = [], torch.zeros(batch_size, dtype=torch.long), torch.zeros(batch_size, dtype=torch.long)
    paths = []
    for i, (sequence, label, p), in enumerate(batch):
        lengths[i], labels[i] = len(sequence), label
        sequences.append(sequence)
        paths.append(p)


    # Combine sequences into a padded matrix
    padded_sequences = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True)
    # Shape: (B x T_max x D) or (B x T_max)

    # If a vector input was given for the sequences, expand (B x T_max) to (B x T_max x 1)
    if padded_sequences.ndim == 2:
        padded_sequences.unsqueeze_(-1)

    return padded_sequences, lengths, labels, paths

class DeepGRU(nn.Module):
    def __init__(self, num_features, num_classes, device=None):
        super(DeepGRU, self).__init__()

        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.num_features = num_features
        self.num_classes = num_classes

        # Encoder
        self.gru1 = nn.GRU(self.num_features, 512, 2, batch_first=True)
        self.gru2 = nn.GRU(512, 256, 2, batch_first=True)
        self.gru3 = nn.GRU(256, 128, 1, batch_first=True)

        # Attention
        self.attention = Attention(128, device=device)

        # Classifier
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(256),
            nn.Dropout(0.5),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
        self.to(device)

    def forward(self, x_padded, x_lengths):
        x_packed = packer(x_padded, x_lengths.cpu(), batch_first=True)

        # Encode
        output, _ = self.gru1(x_packed)
        output, _ = self.gru2(output)
        output, hidden = self.gru3(output)

        # Pass to attention with the original padding
        output_padded, _ = padder(output, batch_first=True)
        attn_output = self.attention(output_padded, hidden[-1:])

        # Classify
        return self.classifier(attn_output)

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ----------------------------------------------------------------------------------------------------------------------
class Attention(nn.Module):
    def __init__(self, attention_dim, device):
        super(Attention, self).__init__()
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.w = nn.Linear(attention_dim, attention_dim, bias=False)
        self.gru = nn.GRU(128, 128, 1, batch_first=True)
        self.to(device)

    def forward(self, input_padded, hidden):
        e = torch.bmm(self.w(input_padded), hidden.permute(1, 2, 0))
        context = torch.bmm(input_padded.permute(0, 2, 1), e.softmax(dim=1))
        context = context.permute(0, 2, 1)

        # Compute the auxiliary context, and concat
        aux_context, _ = self.gru(context, hidden)
        output = torch.cat([aux_context, context], 2).squeeze(1)

        return output




class trinity_Dataset(torch.utils.data.Dataset):
    def __init__(
            self,
            random_start=True,
            subset=0,
            split_number=0,
            representation_type='rep_velocity',
    ):
        """
        """
        X, y = load_rep(representation_type)
        paths = load_rep_info(representation_type)

        max_X = np.max([np.max(x) for x in X])
        X = [np.array([xx / max_X for xx in x]) for x in X]

        self.set = subset
        self.random_start = random_start

        #### build the train subsets: train, test using train_test_split, a stratified split with the labels
        self.X_train, self.X_test, self.y_train, self.y_test, self.paths_train, self.paths_test = get_split(split_number, X, paths)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        #### get the file with index in the corresponding subset
        if self.set == 0:
            matrix = torch.from_numpy(self.X_train[index].astype(np.float64))
            label = torch.tensor(self.y_train[index], dtype=torch.float)
            path = self.paths_train[index]
        else:  # self.set == 1:
            matrix = torch.from_numpy(self.X_test[index].astype(np.float64))
            label = torch.tensor(self.y_test[index], dtype=torch.float)
            path = self.paths_test[index]
        return matrix, label, path


    def get_path(self, index):
        if self.set == 0:
            path = self.paths_train[index]
        else:   # self.set == 1:
            path = self.paths_test[index]
        return path


    def __len__(self):
        if self.set == 0:
            return len(self.X_train)
        else:
            return len(self.X_test)


soft_labels = False


def create_dataset(split_number, representation_type):
    train_dataset = trinity_Dataset(subset=0, split_number=split_number, representation_type=representation_type)
    train_loader = torch.utils.data.DataLoader(train_dataset, collate_fn=collate_fn, batch_size=64, num_workers=4,
                                               pin_memory=True)

    test_dataset = trinity_Dataset(subset=1, split_number=split_number, representation_type=representation_type)
    test_loader = torch.utils.data.DataLoader(test_dataset, collate_fn=collate_fn, batch_size=len(list(test_dataset)),
                                                  num_workers=4, pin_memory=True)

    return train_loader, test_loader


def get_acc(model, device, loader, epoch, verbose=False):
    # Retrieve test set as a single batch
    batch, lengths, labels, paths = next(iter(loader))
    # Send data to device
    batch, lengths, labels = batch.to(device), lengths.to(device), labels.to(device)
    # Calculate predictions for test set
    y = model(batch.float(), lengths)
    y_pred = torch.argmax(y, dim=1).cpu()
    # Calculate accuracy
    acc = balanced_accuracy_score(y_pred=y_pred, y_true=labels.cpu())
    if verbose:
        print('Accuracy: {:.2f}%'.format(acc * 100))
    return y_pred, labels.cpu(), acc


def normalize_list_numpy(list_numpy):
    normalized_list = minmax_scale(list_numpy)
    return normalized_list


def get_rank(model, device, loader, representation_type, verbose=False):
    # Retrieve test set as a single batch
    x, paths = [], []
    for batch, lengths, labels, p in loader:
        # Send data to device
        # print(p)
        batch, labels = batch.to(device), labels.to(device)
        # Calculate predictions for test set
        x_element = model(batch.float(), lengths).cpu().tolist()
        x_element = normalize_list_numpy([[xx[0], xx[1], xx[2]] for xx in x_element])
        x.extend([xx[0] * 0 + xx[1] * 1 + xx[2] * 2 for xx in x_element])
        paths.extend(p)
        # print(np.array(x), np.array(paths))
    # BELA BARTOK
    # pair wise (framewise mean step 1 output, bartok number piece)
    pair_wise = [(m, id2) for m, id2 in zip(x, paths)]
    # sort pieces by piece number
    x = [y[0] for y in pair_wise]
    y = [int(y[1]) for y in pair_wise]
    # compute rho
    sp_bartok = spearmanr(a=x, b=y)
    kb_bartok = kendalltau(x, y, variant='b')
    kc_bartok = kendalltau(x, y, variant='c')
    # HENLE
    # load henle dict
    piece2henle = utils.load_json("mikrokosmos/henle_mikrokosmos.json")
    # create triplet wise (mean ids and henle)
    pair_wise = [(m, id2, piece2henle[str(id2)]) for m, id2 in pair_wise]
    # order by mean (1st index) and piece number (2nd index)
    y = [int(y[2]) for y in pair_wise]
    # compute rho
    sp_henle = spearmanr(a=x, b=y)
    kb_henle = kendalltau(x, y, variant='b')
    kc_henle = kendalltau(x, y, variant='c')
    element = {
        'sp_bartok': sp_bartok,
        'kb_bartok': kb_bartok,
        'kc_bartok': kc_bartok,
        'sp_henle': sp_henle,
        'kb_henle': kb_henle,
        'kc_henle': kc_henle,
    }
    return element


def classification(train_loader,  test_loader, split_number=0,
                   soft_labels=False,
                   save=True, verbose=True, plot=False, representation_type="rep_velocity"):
    # Create a DeepGRU neural network model
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    n_features = 88 if representation_type == "rep_note" else 10
    n_grades = 3
    model = DeepGRU(n_features, n_grades, device=None)

    # Set loss function and optimizer
    if soft_labels:
        criterion = torch.nn.BCELoss()
    else:
        criterion = torch.nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)

    label_padder = torch.nn.ZeroPad2d((1, 1, 0, 0))

    def label_softener(lbl):
        highergrade = 0.3 * label_padder(lbl[:, :-2])
        lowergrade = 0.3 * label_padder(lbl[:, 2:])
        lbl = lbl + highergrade + lowergrade
        return lbl / 1.6

    # Toggle evaluation mode
    model.eval()

    n_epoch = 20
    for epoch in tqdm(range(1, n_epoch+1), desc='Epoch'):
        # Toggle training mode
        model.train()
        # Training loop
        for batch, lengths, labels, paths in tqdm(train_loader, desc='Training batch', leave=False):
            # Send data to the device
            batch, lengths, labels = batch.to(device), lengths.to(device), labels.to(device)

            # Reset the optimizer
            optimizer.zero_grad()

            # Calculate predictions for batch
            log_prob = model(batch.float(), lengths)

            # Calculate and back-propagate loss
            if soft_labels:
                labels = F.one_hot(labels, num_classes=n_grades).float()
                labels = label_softener(labels)
                prob = log_prob.exp()
                prob = prob / prob.sum()
                loss = criterion(prob.float(), labels.float())
            else:
                loss = criterion(log_prob.float(), labels)

            # y_pred = torch.argmax(log_prob, dim=1)
            loss.backward()

            # Update the optimizer
            optimizer.step()


        if save and epoch in [20]:
            if not os.path.exists(f"results/deepgru/results_{representation_type}.json"):
                save_json({}, f"results/deepgru/results_{representation_type}.json")
            results = load_json(f"results/deepgru/results_{representation_type}.json")
            _, _, train_balanced_acc = get_acc(model, device, train_loader, epoch=epoch, verbose=verbose)
            _, _, test_balanced_acc = get_acc(model, device, test_loader, epoch=epoch, verbose=verbose)
            results[f"{split_number}:{epoch}"] = {
                'train_balanced_acc': train_balanced_acc,
                'test_balanced_acc': test_balanced_acc,
                "epoch": epoch
            }
            save_json(results, f"results/deepgru/results_{representation_type}.json")
            if epoch in [20]:
                torch.save(model.state_dict(),
                           f'results/deepgru/split:{split_number}_epoch:{epoch}_{representation_type}.pkl')


    if plot:
        y_pred, _ = get_acc(model, device, test_loader, epoch='last', verbose=verbose)
        # Calculate confusion matrix
        classes = range(n_grades)
        cm = confusion_matrix(labels.cpu(), y_pred.cpu(), labels=classes)

        # Display accuracy and confusion matrix
        labels = range(n_grades)

        df = pd.DataFrame(cm, index=labels, columns=labels)
        plt.figure(figsize=(7, 7))
        sns.heatmap(df, annot=True, cbar=False)
        plt.title(f'Confusion matrix for test set predictions rep{representation_type}', fontsize=14)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.show()


def attention(rang=50, representation_type="rep_velocity"):
    for split_number in range(rang):
        print(f"SPLIT NUMBER {split_number}")
        train_loader, test_loader = create_dataset(split_number=split_number, representation_type=representation_type)
        classification(train_loader, test_loader, split_number=split_number, representation_type=representation_type)


def measure_ranking(train_loader, test_loader, split_number, representation_type):
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    n_features = 88 if representation_type == "rep_note" else 10
    n_grades = 3
    model = DeepGRU(num_features=n_features, num_classes=n_grades)
    model.load_state_dict(torch.load(f'results/deepgru/split:{split_number}_epoch:20_{representation_type}.pkl',
                          map_location=torch.device(device=device)))
    element = get_rank(model, device, test_loader, representation_type=representation_type, verbose=False)

    if not os.path.exists(f"results/deepgru/rank_results_{representation_type}.json"):
        save_json({}, f"results/deepgru/rank_results_{representation_type}.json")
    results = load_json(f"results/deepgru/rank_results_{representation_type}.json")
    results[f"{split_number}:{20}"] = element
    save_json(results, f"results/deepgru/rank_results_{representation_type}.json")


def rank_attention(rang=50, representation_type="rep_velocity"):
    for split_number in range(rang):
        print(f"SPLIT NUMBER {split_number}")
        train_loader, test_loader = create_dataset(split_number=split_number, representation_type=representation_type)
        measure_ranking(train_loader, test_loader, split_number=split_number, representation_type=representation_type)


if __name__ == '__main__':
    if not os.path.exists("results/deepgru"):
        os.mkdir("results/deepgru")
    # rank_attention(50, representation_type="rep_velocity")
    # rank_attention(50, representation_type="rep_prob")
    # rank_attention(50, representation_type="rep_finger_nakamura")
    # rank_attention(50, representation_type="rep_note")
    # rank_attention(50, representation_type="rep_finger")

    attention(50, representation_type="rep_d_nakamura")
    rank_attention(50, representation_type="rep_d_nakamura")

    # t= trinity_Dataset()
    # print(t.paths_train)

