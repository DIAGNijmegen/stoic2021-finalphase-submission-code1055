import torch
import torchvision
import numpy as np
import os
import random
from data.mosmed import get_mosmed_dataset
from config.dataconfig import MosmedConfig, HustConfig
from data.hust import get_hust_dataset


class SupervisedDataset(torch.utils.data.Dataset):
    def __init__(self, path = '/data/ssd1/kienzlda/data_stoic/segmentation/resized256', transform=None, train=True):
        path = os.path.join(path, 'Train')
        self.path = path
        self.transform = transform

        filenamelist = []
        dataname_old = None
        for dataname in sorted(os.listdir(path)):
            dataname = os.path.join(path, dataname)
            if dataname_old == None:
                dataname_old = dataname
            else:
                filenamelist.append([dataname_old, dataname])
                dataname_old = None
        train_len = int(0.85*len(filenamelist))
        val_len = len(filenamelist) - train_len
        rand = random.Random(42)
        rand.shuffle(filenamelist)
        self.filenamelist = filenamelist[:train_len] if train else filenamelist[train_len:]

    def __len__(self):
        return len(self.filenamelist)

    def __getitem__(self, idx):
        img, label = self.filenamelist[idx]
        img, label = np.load(img).astype(np.float32), np.load(label).astype(np.float32)
        if self.transform:
            img, label = self.transform(img, label)
        #img, label = np.expand_dims(img, axis=0), np.expand_dims(label, axis=0)
        img, label = img.unsqueeze(0), label.unsqueeze(0)
        return img, label


class ValidationDataset(torch.utils.data.Dataset):
    def __init__(self, path = '/data/ssd1/kienzlda/data_stoic/segmentation/resized256', transform=None):
        path = os.path.join(path, 'Validation')
        self.path = path
        self.transform = transform

        filenamelist = [os.path.join(path, dataname) for dataname in sorted(os.listdir(path))]
        self.filenamelist = filenamelist

    def __len__(self):
        return len(self.filenamelist)

    def __getitem__(self, idx):
        img = self.filenamelist[idx]
        id = img.split(os.path.join(self.path, 'volume-covid19-A-0'))[-1].strip('_ct.npy')
        img = np.load(img).astype(np.float32)
        if self.transform:
            img, __ = self.transform(img, None)
        #img = np.expand_dims(img, axis=0)
        img = img.unsqueeze(0)
        return img, id


class UnsupervisedDataset_stoic(torch.utils.data.Dataset):
    def __init__(self, path = '/data/ssd1/kienzlda/data_stoic/segmentation/resized256', transform=None):
        path = os.path.join(path, 'unsupervised_stoic')
        self.path = path
        self.transform = transform

        filenamelist = [os.path.join(path, dataname) for dataname in sorted(os.listdir(path))]
        self.filenamelist = filenamelist

    def __len__(self):
        return len(self.filenamelist)

    def __getitem__(self, idx):
        img = self.filenamelist[idx]
        img = np.load(img).astype(np.float32)
        if self.transform:
            img, __ = self.transform(img, None)
        #img = np.expand_dims(img, axis=0)
        img = img.unsqueeze(0)
        return img

class UnsupervisedDataset_tcia(torch.utils.data.Dataset):
    def __init__(self, path = '/data/ssd1/kienzlda/data_stoic/segmentation/resized256', transform=None):
        path = os.path.join(path, 'unsupervised_tcia')
        self.path = path
        self.transform = transform

        filenamelist = [os.path.join(path, dataname) for dataname in sorted(os.listdir(path))]
        self.filenamelist = filenamelist

    def __len__(self):
        return len(self.filenamelist)

    def __getitem__(self, idx):
        img = self.filenamelist[idx]
        #id = img.split(os.path.join(self.path, 'volume-covid19-A-0'))[-1].strip('.npy')
        img = np.load(img).astype(np.float32)
        #img = np.expand_dims(img, axis=0)
        if self.transform:
            img, __ = self.transform(img, None)
        img = img.unsqueeze(0)
        return img#, id

class MosMedDataset(torch.utils.data.Dataset):
    def __init__(self, config, mode = 'train'):
        dataconfig_mos = MosmedConfig(config)
        dataconfig_mos.do_normalization = config.DO_NORMALIZATION if hasattr(config, 'DO_NORMALIZATION') else dataconfig_mos.do_normalization
        dataset = get_mosmed_dataset(dataconfig_mos)
        idxs = [i for i in range(len(dataset))]
        rand = random.Random(42)
        rand.shuffle(idxs)
        self.idxs = idxs[:int(0.8*len(dataset))] if mode == 'train' else idxs[int(0.8*len(dataset)):]
        self.dataset = dataset

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, idx):
        return self.dataset[self.idxs[idx]]

class HustDataset(torch.utils.data.Dataset):
    def __init__(self, config, mode = 'train'):
        dataconfig_hust = HustConfig(config)
        dataconfig_hust.do_normalization = config.DO_NORMALIZATION if hasattr(config, 'DO_NORMALIZATION') else dataconfig_hust.do_normalization
        dataset = get_hust_dataset(dataconfig_hust)
        idxs = [i for i in range(len(dataset))]
        rand = random.Random(42)
        rand.shuffle(idxs)
        self.idxs = idxs[:int(0.8*len(dataset))] if mode == 'train' else idxs[int(0.8*len(dataset)):]
        self.dataset = dataset

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, idx):
        return self.dataset[self.idxs[idx]]






if __name__ == '__main__':
    SDS = UnsupervisedDataset_stoic()
    print(len(SDS))
    supervised_loader = torch.utils.data.DataLoader(SDS, batch_size=1)

    for img in supervised_loader:
        print(img.shape)
        #print(label.shape)
        break