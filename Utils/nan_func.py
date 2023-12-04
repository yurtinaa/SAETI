from dataclasses import dataclass, MISSING
import copy
from tqdm import tqdm
import torch
import wandb
import numpy as np


def last_add_nan(x, percent=0.25, init=0.0):
    X = x.clone()
    X[X != X] = init
    X[:, -1] = np.nan
    return X


def all_random_add_nan(x, percent=0.25):
    X = x.clone()
    #    print(X.shape)
    # y = self.y[idx].clone()
    # if self.type_nan == 'last':
    # X[-1, :] = np.nan

    index = (torch.rand(size=x.shape) < percent).bool()
    #  print(index.shape, X.shape)
    if len(index.shape) == 1:
        index = index[:, None]

    X[:, 1:-1, :][index[:, 1:-1, :]] = np.nan
    return X


def last_random_add_nan(x, percent=0.25):
    X = x.clone()
    # print(X.shape)
    # y = self.y[idx].clone()
    # if self.type_nan == 'last':
    # X[-1, :] = np.nan

    index = (torch.rand(size=x.shape) < percent * 2).bool()
    #  print(index.shape, X.shape)
    if len(index.shape) == 1:
        index = index[:, None]
    half_len = x.shape[1] // 2
    X[:, half_len:-1, :][index[:, half_len:-1, :]] = np.nan
    return X


def batch_last_add_nan(x, percent=0.125):
    X = x.clone()
    #    print(X.shape)
    # y = self.y[idx].clone()
    # if self.type_nan == 'last':
    # X[-1, :] = np.nan

    index = (torch.rand(size=x[:, x.shape[1] // 2:].shape[1:]) < percent).bool()
    index = index.unsqueeze(0)
    index = index.repeat(X.shape[0], 1, 1)
    #  print(index.shape, X.shape)
    if len(index.shape) == 1:
        index = index[:, None]

    X[:, x.shape[1] // 2:-1, :][index[:, :-1, :]] = np.nan
    return X


def batch_random_add_nan(x, percent=0.1):
    X = x.clone()
    #  print('pecent:', percent)
    index = (torch.rand(size=X.shape[1:]) < percent).bool()
    index = index.unsqueeze(0)
    #   print(index.shape, X.shape)
    #   print(batch.shape[0])
    index = index.repeat(X.shape[0], 1, 1)
    index[:, 0] = False
    index[:, -1] = False
    X[index] = np.nan

    print((X[index] != X[index]).sum())

    return X


dict_func_nan = {
    'random': all_random_add_nan,
    'batch': batch_random_add_nan,
    'last': last_add_nan,
    'batch_last': batch_last_add_nan,
    'random_last': last_random_add_nan,
}
