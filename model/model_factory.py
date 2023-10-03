import torch.nn
from torch import nn
import toml

#####################################
#
# Common utils
#
#####################################
def loadconfig(args):

    if args.method == "SingleDNN":
        args.config =  toml.load('config/SingleDNN.toml')



def ObtainModel(args):

    if args.method == "SingleDNN":
        obj = singleDNN(args)

    return obj

#####################################
#
# DNN models
#
#####################################

class singleDNN(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.args = args

    def build_layer(self):

        layers = []
        input_dim = 26
        layers.append(torch.nn.Linear(26, ))
        layers.append(torch.nn.ReLU)





class MultiDNN(nn.Module):
    pass


class MultiKmer(nn.Module):
    pass

#####################################
#
# CNN models
#
#####################################
class singleCNN(nn.Module):
    pass


class CNN_DNN(nn.Module):
    pass
