import random
import numpy as np
import torch
from train.TrainMutliDNN import *
from train.Train_ResiDNN import *

def set_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_model(args, dataset, model):

    if args.method == "MutliDNN":
        train_MutliDNN(args, dataset, model)

    if args.method == "ResiDNN":
        train_ResiDNN(args, dataset, model)


