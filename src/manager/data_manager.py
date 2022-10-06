"""

* prepare and preprocess datasets
* make_loader returns data loader in PyTorch

"""

import cv2
import glob
import random
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler

import torch
from torch.utils.data import TensorDataset, DataLoader


class DataManager:
    def __init__(self, cfg, path,):
        self.root = str(path.parent.absolute())
        self.name = cfg.dataset
        self.template = cfg.name_template
        self.seed = cfg.seed
        self.batch_size = cfg.batch_size

    def get_data(self,):
        le = LabelEncoder()
        if self.name == 'TCGA':
            self.out_dim = 5
            self.in_dim = 20531
            X = pd.read_csv(self.root + '/datasets/TCGA/data.csv', index_col=0).values
            y = pd.read_csv(self.root + '/datasets/TCGA/label.csv', index_col=0).values.reshape(-1)

        elif self.name == 'madelon':
            self.out_dim = 2
            self.in_dim = 500
            X = pd.read_csv(self.root + '/datasets/madelon/data.csv', index_col=0).values
            y = pd.read_csv(self.root + '/datasets/madelon/label.csv', index_col=0).values.reshape(-1)

        elif self.name == 'ringnorm':
            self.out_dim = 2
            self.in_dim = 20
            X = pd.read_csv(self.root + '/datasets/ringnorm/data.csv', index_col=0).values
            y = pd.read_csv(self.root + '/datasets/ringnorm/label.csv', index_col=0).values.reshape(-1)
            
        elif self.name == 'relathe':
            self.out_dim = 2
            self.in_dim = 4322
            X = pd.read_csv(self.root + '/datasets/relathe/data.csv', index_col=0).values
            y = pd.read_csv(self.root + '/datasets/relathe/label.csv', index_col=0).values.reshape(-1)

        elif self.name == 'isolet':
            self.out_dim = 26
            self.in_dim = 617
            X = pd.read_csv(self.root + '/datasets/isolet/data.csv', index_col=0).values
            y = pd.read_csv(self.root + '/datasets/isolet/label.csv', index_col=0).values.reshape(-1)

        else:
            print('Invalid Dataset')
            exit(1)

        y = le.fit_transform(y)
        print(X.shape, y.shape)
        return X, y

    def get_info(self):
        return self.in_dim, self.out_dim

    def get_templates(self,):
        p = glob.glob(self.root + f'/template/{self.template}/*')
        p.sort()
        templates = [cv2.imread(f, cv2.IMREAD_GRAYSCALE) for f in p]
        for i in range(len(templates)):
            templates[i] = templates[i]/255
        return templates

    def preprocess(self, X_train, X_test, sc=False):
        scaler = StandardScaler() if sc else MinMaxScaler()
        X_train = scaler.fit_transform(X_train)
        if not sc:
            X_test = scaler.transform(X_test).clip(0, 1)
        else:
            X_test = scaler.transform(X_test)
        return X_train, X_test

    def make_loader(self, X_train, y_train, X_test=None, y_test=None, use_template=True):
        X_train = torch.from_numpy(X_train).float().contiguous()
        y_train = torch.from_numpy(y_train).long()
        if X_test is not None and y_test is not None:
            X_test = torch.from_numpy(X_test).float().contiguous()
            y_test = torch.from_numpy(y_test).long()

        temp_train, temp_test = None, None
        if use_template:
            templates = self.get_templates()
            temp_train = [templates[i] for i in y_train]
            temp_train = torch.from_numpy(np.array(temp_train)).float().contiguous()
            if y_test is not None:
                temp_test = [templates[i] for i in y_test]
                temp_test = torch.from_numpy(np.array(temp_test)).float().contiguous()

        return self._make_loader(
                                   X_train=X_train,
                                   y_train=y_train,
                                   temp_train=temp_train,
                                   X_test=X_test,
                                   y_test=y_test,
                                   temp_test=temp_test,
                                )

    def _make_loader(self,
                     X_train,
                     y_train,
                     temp_train,
                     X_test,
                     y_test,
                     temp_test):

        g = torch.Generator()
        g.manual_seed(self.seed)

        if temp_train is not None:
            trainset = TensorDataset(X_train, y_train, temp_train)
            train_loader = DataLoader(trainset,
                                      batch_size=self.batch_size,
                                      shuffle=True,
                                      drop_last=True,
                                      worker_init_fn=seed_worker,
                                      generator=g,)
            test_loader = None
            if X_test is not None:
                testset = TensorDataset(X_test, y_test, temp_test)
                test_loader = DataLoader(testset, batch_size=self.batch_size)
        else:
            trainset = TensorDataset(X_train, y_train)
            train_loader = DataLoader(trainset,
                                      batch_size=self.batch_size,
                                      shuffle=True,
                                      drop_last=True,
                                      worker_init_fn=seed_worker,
                                      generator=g,)
            test_loader = None
            if X_test is not None:
                testset = TensorDataset(X_test, y_test)
                test_loader = DataLoader(testset, batch_size=self.batch_size)

        return train_loader, test_loader


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
