import os
import requests

import numpy as np

import torch
from torch.utils.data import DataLoader, Subset, random_split

import torchvision
from torchvision import transforms
from sklearn.model_selection import train_test_split


def get_cifar10_data():
    """Returns a dictionary containing all CIFAR10 data loaders. 
    """
    normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

    train_set = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=normalize)
    # split train set into a retain set and a compact train set
    train_labels_split = np.zeros((len(train_set),))
    for i in range(len(train_set)):
        train_labels_split[i] = train_set[i][1]
    retain_train_idx, compact_train_idx = train_test_split(
        np.arange(len(train_set)), test_size=512, shuffle=True, stratify=train_labels_split)
    np.save("retain_train_idx.npy", retain_train_idx)
    np.save("compact_train_idx.npy", compact_train_idx)
    retain_train_set = Subset(train_set, retain_train_idx)
    compact_train_set = Subset(train_set, compact_train_idx)
    
    # we split held out data into test and validation set
    test_set = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=normalize)
    test_labels_split = np.zeros((len(test_set),))
    for i in range(len(test_set)):
        test_labels_split[i] = test_set[i][1]
    retain_test_idx, compact_test_idx = train_test_split(
        np.arange(len(test_set)), test_size=512, shuffle=True, stratify=test_labels_split)
    np.save("retain_test_idx.npy", retain_test_idx)
    np.save("compact_test_idx.npy", compact_test_idx)
    retain_test_set = Subset(test_set, retain_test_idx)
    compact_test_set = Subset(test_set, compact_test_idx)

    retain_train_loader = DataLoader(
        retain_train_set, batch_size=128, shuffle=True)
    compact_train_loader = DataLoader(
        compact_train_set, batch_size=128, shuffle=True)
    
    retain_test_loader = DataLoader(
        retain_test_set, batch_size=128, shuffle=False)
    compact_test_loader = DataLoader(
        compact_test_set, batch_size=128, shuffle=False)
    
    return {
        "retain_train": retain_train_loader,
        "compact_train": compact_train_set,
        "retain_test": retain_test_loader,
        "compact_test": compact_test_loader
    }