import random

import numpy as np
import torch

from torch.utils.data import Dataset, SubsetRandomSampler, DataLoader

#####################################
#
# Common utils
#
#####################################
class CommonDataset(Dataset):
    def __init__(self, X, y):
        self.x_data =  X
        self.y_data = y

        self.length = len(self.y_data)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.length


class DataLoad(object):

    def __init__(self, args):

        self.args = args

        print("loading data...")
        self.wt = self.preprocess(args.mod, args.motif)
        self.ko = self.preprocess(args.unmod, args.motif)
        print("finish!")

        print("reshape data for specific model...")
        self.reshapeData()
        print("finish!")

    def preprocess(self, path, motif):

        cur_info = []
        mat_info = []
        read_info = []

        for i in open(path, "r"):

            ele = i.rstrip().split()
            if ele[2] == motif:
                # id
                id = [ele[0], ele[2], ele[9]]
                read_info.append(id)

                # current signal
                cur_mean = [float(item) for item in ele[3].split("|")]
                cur_std = [float(item) for item in ele[4].split("|")]
                cur_median = [float(item) for item in ele[5].split("|")]
                cur_length = [int(item) for item in ele[6].split("|")]
                cur = np.stack([cur_mean, cur_std, cur_median, cur_length])
                cur_info.append(cur)

                # matching
                base, strand, cov, q_mean, q_median, q_std, mis, ins, deli = ele[8].split("|")
                mat = [float(item) for item in [q_mean, q_median, q_std, mis, ins, deli]]
                mat_info.append(mat)

        return {'read':read_info, 'current':cur_info, 'matching':mat_info}

    def reshapeData(self):

        nums = min(len(self.wt['read']), len(self.ko['read']))
        ratio = self.args.ratio # wt:ko
        ratio = [int(item) for item in  ratio.split(":")]

        if ratio[0]/ratio[1]>=1:
            wt_nums = nums
            ko_nums = int(np.floor(nums*ratio[1]/ratio[0]))
        else:
            ko_nums = nums
            wt_nums = int(np.floor(nums*ratio[1]/ratio[0]))

        wt_indice = random.sample(range(0, wt_nums), wt_nums)
        ko_indice =  random.sample(range(0, wt_nums), ko_nums)

        wt = np.concatenate([np.stack([np.concatenate(item) for item in self.wt['current']]), np.stack(self.wt['matching'])], axis=1)
        ko = np.concatenate([np.stack([np.concatenate(item) for item in self.ko['current']]), np.stack(self.ko['matching'])], axis=1)

        wt = wt[wt_indice]
        ko = ko[ko_indice]

        X = np.concatenate([wt, ko], axis=0)
        y = np.concatenate([np.zeros(len(wt)), np.ones(len(ko))], axis=0)

        shuffle = random.sample(range(0, len(y)), len(y))
        X = torch.FloatTensor(X[shuffle])
        y = torch.FloatTensor(y)

        # construct dataset
        Dataset = CommonDataset(X, y)

        # split dataset for train, validate and test
        dataset_size = len(Dataset)
        indices = random.sample(range(0, dataset_size), dataset_size)

        test_size = int(0.2 * dataset_size)
        valid_size = int(0.2 * dataset_size)
        train_size = int(0.6 * dataset_size)

        train_indices = indices[:train_size]
        val_indices = indices[train_size:(train_size+valid_size)]
        test_indices = indices[(train_size+valid_size):]

        train_sampler = SubsetRandomSampler(train_indices)
        val_sampler = SubsetRandomSampler(val_indices)
        test_sampler = SubsetRandomSampler(test_indices)

        batch_size = self.args.config['train']['batch_size']
        self.train_loader = DataLoader(Dataset, batch_size=batch_size, sampler=train_sampler)
        self.val_loader = DataLoader(Dataset, batch_size=batch_size, sampler=val_sampler)
        self.test_loader = DataLoader(Dataset, batch_size=batch_size, sampler=test_sampler)