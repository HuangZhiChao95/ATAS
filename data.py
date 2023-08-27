
import torch
import torchvision
from torchvision import transforms
from torchvision import datasets
import os
import torch.utils.data
import random

ROOT = "/path/to/data/"
shapes_dict = {'cifar10': [3, 40, 40], "cifar100": [3, 40, 40], "imagenet": [3, 32, 32]}
datasets_dict = {'cifar10': datasets.CIFAR10, "cifar100": datasets.CIFAR100, "imagenet": datasets.ImageFolder}
dir_dict = {'cifar10': "CIFAR10", 'cifar100': "CIFAR100", "imagenet": "imagenet"}
cls_dict = {'cifar10': 10, 'cifar100': 100, "imagenet": 1000}

class IndexDataset(torch.utils.data.Dataset):

    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, idx):
        return self.dataset[idx] + (idx,)

    def __len__(self):
        return len(self.dataset)


def load_data(dataset, batch_size, num_workers):
    assert dataset != 'imagenet'
    transform = transforms.Compose([transforms.Pad(padding=4), transforms.ToTensor()])
    dir_ = os.path.join(ROOT, dir_dict[dataset])
    dataset_f = datasets_dict[dataset]

    train_data = dataset_f(dir_, train=True, transform=transform, download=True)

    train_data = IndexDataset(train_data)

    test_data = dataset_f(dir_, train=False, transform=transform, download=True)

    train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=num_workers)

    return train_loader, test_loader, len(train_data)


def load_data_imagenet(dataset, batch_size, num_workers):
    assert dataset == 'imagenet'
    train_transform = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
    test_transform = transforms.Compose([transforms.Resize((256, 256)), transforms.CenterCrop(224), transforms.ToTensor()])

    dir_ = os.path.join(ROOT, dir_dict[dataset])
    dataset_f = datasets_dict[dataset]

    train_data = dataset_f(os.path.join(dir_, "train"), transform=train_transform)
    train_data = IndexDataset(train_data)
    test_data = dataset_f(os.path.join(dir_, "val"), transform=test_transform)

    train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=num_workers)

    return train_loader, test_loader, len(train_data)


def load_data_imagenet_val(dataset, batch_size, num_workers):
    assert dataset == 'imagenet'
    test_transform = transforms.Compose([transforms.Resize((256, 256)), transforms.CenterCrop(224), transforms.ToTensor()])

    dir_ = os.path.join(ROOT, dir_dict[dataset])
    dataset_f = datasets_dict[dataset]

    test_data = dataset_f(os.path.join(dir_, "val"), transform=test_transform)

    test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=num_workers)

    return test_loader





