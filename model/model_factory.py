import torch.nn
from torch import nn
import toml

#####################################
#
# Common utils
#
#####################################
def loadconfig(args):

    if args.method == "MutliDNN":
        args.config =  toml.load('config/MutliDNN.toml')
        RRACH =  ["AAACA", "AAACT", "AGACC", "GAACA", "GAACT", "GGACC", "AAACC", "AGACA", "AGACT", "GAACC", "GGACA",
                "GGACT"]
        # args.motif = "GGACT"

def ObtainModel(args):

    if args.method == "MutliDNN":
        obj = MutliDNN(args)

    return obj


#####################################
#
# DNN models
#
#####################################

class MutliDNN(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.args = args

        # build model
        self.site_model = self.ExtractSite()
        self.read_model = self.ExtractRead()
        self.site_pre = self.site_pre_model()
        self.read_pre = self.read_pre_model()

    def build_layer(self, dims, pre=None):

        layers = []

        for index, item in enumerate(dims):
            input_channel = item[0]
            output_channel = item[1]
            layers.append(torch.nn.Linear(input_channel, output_channel))
            if pre=="Sigmoid" and index==(len(dims)-1):
                layers.append(torch.nn.Sigmoid())
            else:
                layers.append(torch.nn.ReLU())

        return layers
    def ExtractSite(self):


        dims = self.args.config['block']['ExtractSite']['dims']
        site_model = nn.Sequential(*self.build_layer(dims))

        return site_model

    def ExtractRead(self):

        dims = self.args.config['block']['ExtractRead']['dims']
        read_model = nn.Sequential(*self.build_layer(dims))

        return read_model

    def site_pre_model(self):

        dims = self.args.config['block']['site_pre_model']['dims']
        activation = self.args.config['block']['site_pre_model']['activation']
        site_pre_model =  nn.Sequential(*self.build_layer(dims, pre=activation))

        return site_pre_model

    def read_pre_model(self):

        dims = self.args.config['block']['read_pre_model']['dims']
        activation = self.args.config['block']['read_pre_model']['activation']
        read_pre_model =  nn.Sequential(*self.build_layer(dims, pre=activation))

        return read_pre_model
    def forward(self, S, X):

        # site information extract
        site_nn = self.site_model(S)

        # read information extract
        read_nn = self.read_model(X)

        # concate read and site information
        ## read->site
        readTosite = torch.concatenate([S, torch.squeeze(read_nn)], dim=1)

        ## site->read
        siteToread = torch.concatenate([X, torch.reshape(site_nn,  (-1, 20,20))], dim=2)

        # readTosite predict
        site_predict = self.site_pre(readTosite)

        # siteToread repdict
        read_predict = self.read_pre(siteToread)
        read_predict_site = 1 - torch.prod(1 - read_predict, axis=1)

        return site_predict, read_predict_site


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
