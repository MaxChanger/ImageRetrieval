import torch
import numpy as np
import copy
import re

is_torchvision_installed = True
try:
    import torchvision
except:
    is_torchvision_installed = False
import torch.utils.data
import random


def find_all_index(arr, item):
    return [i for i, a in enumerate(arr) if a == item]


class BalancedBatchSampler(torch.utils.data.sampler.Sampler):
    def __init__(self, dataset, labels=None, type='single_label'):
        self.labels = labels
        self.dataset = dict()
        self.balanced_max = 0
        # Save all the indices for all the classes
        if type == 'single_label':
            for idx in range(0, len(dataset)):
                l = self._get_label(dataset, idx)
                label = str(l)
                if label not in self.dataset:
                    self.dataset[label] = list()
                self.dataset[label].append(idx)
                self.balanced_max = len(self.dataset[label]) \
                    if len(self.dataset[label]) > self.balanced_max else self.balanced_max
        elif type == 'multi_label':
            for idx in range(0, len(dataset)):
                label = self._get_label(dataset, idx)
                label_index = find_all_index(label, 1)
                label_temp = np.zeros((len(label_index), len(label))).astype(int)
                for i in range(len(label_index)):
                    label_temp[i, label_index[i]] = 1

                for l in label_temp:
                    label = str(l)
                    if label not in self.dataset:
                        self.dataset[label] = list()
                    self.dataset[label].append(idx)
                    self.balanced_max = len(self.dataset[label]) \
                        if len(self.dataset[label]) > self.balanced_max else self.balanced_max

        # Oversample the classes with fewer elements than the max
        self.dataset_b = copy.deepcopy(self.dataset)
        for label in self.dataset:
            while len(self.dataset[label]) < self.balanced_max:
                self.dataset[label].append(random.choice(self.dataset[label]))
        self.keys = list(self.dataset.keys())
        self.currentkey = 0
        self.indices = [-1] * len(self.keys)
        print(self.balanced_max, self.keys)

    def __iter__(self):
        while self.indices[self.currentkey] < round(0.01*self.balanced_max)  - 1:
            self.indices[self.currentkey] += 1
            yield self.dataset[self.keys[self.currentkey]][self.indices[self.currentkey]]
            self.currentkey = (self.currentkey + 1) % len(self.keys)

        for label in self.dataset_b:
            self.dataset[label] = []
            while len(self.dataset[label]) < self.balanced_max:
                self.dataset[label].append(random.choice(self.dataset_b[label]))
            print(self.dataset[label])
        self.keys = list(self.dataset.keys())
        self.currentkey = 0
        self.indices = [-1] * len(self.keys)

    def _get_label(self, dataset, idx, labels=None):
        if self.labels is not None:
            return self.labels[idx].item()
        else:
            # Trying guessing
            dataset_type = type(dataset)
            if is_torchvision_installed and dataset_type is torchvision.datasets.MNIST:
                return dataset.train_labels[idx].item()
            elif is_torchvision_installed and dataset_type is torchvision.datasets.ImageFolder:
                return dataset.imgs[idx][1]
            elif 'tag' in dataset.df.columns:
                label_str = dataset.df.tag[idx]
            # elif 'Shape' in dataset.df.columns:
            #     label_str = dataset.df.Shape[idx]
            # elif 'Margin' in dataset.df.columns:
            #     label_str = dataset.df.Margin[idx]
                # label_ls = [int(i) for i in re.findall("\d+", label_str)]
                try:
                    label_ls = int(label_str)
                except:
                    print(dataset.df.crop_512_path[idx])
                return label_ls
            else:
                raise Exception("You should pass the tensor of labels to the constructor as second argument")

    def __len__(self):
        return self.balanced_max * len(self.keys)