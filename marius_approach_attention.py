import os
import json

import numpy as np
import pandas as pd
import sklearn
import torch
from matplotlib import pyplot as plt
from sequentia import Standardize, KNNClassifier, GMMHMM, HMMClassifier
from sklearn.metrics import confusion_matrix, balanced_accuracy_score
from sklearn.model_selection import train_test_split
import seaborn as sns
from tqdm import tqdm
from sequentia.classifiers.rnn import collate_fn
from sklearn import preprocessing
import torch.nn.functional as F

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence as packer, pad_packed_sequence as padder


# ----------------------------------------------------------------------------------------------------------------------
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
        else:  # self.set == 1:
            matrix = torch.from_numpy(self.X_test[index].astype(np.float64))
            label = torch.tensor(self.y_test[index], dtype=torch.float)
        return matrix, label

    def __len__(self):
        if self.set == 0:
            return len(self.X_train)
        else:
            return len(self.X_test)


soft_labels = False


def create_dataset(split_number):
    train_dataset = trinity_Dataset(subset=0, split_number=split_number)
    train_loader = torch.utils.data.DataLoader(train_dataset, collate_fn=collate_fn, batch_size=64, num_workers=4,
                                               pin_memory=True)

    test_dataset = trinity_Dataset(subset=1, split_number=split_number)
    test_loader = torch.utils.data.DataLoader(test_dataset, collate_fn=collate_fn, batch_size=len(list(test_dataset)),
                                                  num_workers=4, pin_memory=True)

    return train_loader, test_loader


def get_acc(model, device, loader, epoch, verbose=False):
    # Retrieve test set as a single batch
    batch, lengths, labels = next(iter(loader))
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


def classification(train_loader,  test_loader, split_number=0,
                   soft_labels=False, n_features=10, n_grades=3,
                   save=True, verbose=True, plot=False):
    # Create a DeepGRU neural network model
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
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

    n_epoch = 200
    for epoch in tqdm(range(n_epoch), desc='Epoch'):
        # Toggle training mode
        model.train()
        total_loss = 0
        # Training loop
        for batch, lengths, labels in tqdm(train_loader, desc='Training batch', leave=False):
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
                total_loss += loss

            # y_pred = torch.argmax(log_prob, dim=1)
            loss.backward()

            # Update the optimizer
            optimizer.step()


        if save and epoch in [0, 10, 25, 50, 75, 100, 150, 199]:
            if not os.path.exists("results/deepgru/results.json"):
                save_json({}, "results/deepgru/results.json")
            results = load_json("results/deepgru/results.json")
            _, _, train_balanced_acc = get_acc(model, device, train_loader, epoch=epoch, verbose=verbose)
            _, _, test_balanced_acc = get_acc(model, device, test_loader, epoch=epoch, verbose=verbose)
            results[f"{split_number}:{epoch}"] = {
                'train_balanced_acc': train_balanced_acc,
                'test_balanced_acc': test_balanced_acc,
                "epoch": epoch
            }
            save_json(results, "results/deepgru/results.json")
            if epoch in [25]:
                torch.save(model.state_dict(), f'results/deepgru/split:{split_number}_epoch:{epoch}.pkl')

        print_loss = total_loss / len(train_loader)
        print(f"Epoch: {epoch} | val_loss: {print_loss}")


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
        plt.title('Confusion matrix for test set predictions', fontsize=14)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.show()




def attention(rang=50):
    for split_number in range(rang):
        print(f"SPLIT NUMBER {split_number}")
        train_loader, test_loader = create_dataset(split_number=split_number)
        classification(train_loader, test_loader, split_number=split_number)


if __name__ == '__main__':
    if not os.path.exists("results/deepgru"):
        os.mkdir("results/deepgru")
    attention(1)