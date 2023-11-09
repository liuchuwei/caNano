import torch.nn
from torch import nn
import toml

#####################################
#
# Common utils
#
#####################################
def loadconfig(args):

    # if args.method == "ResiDNN":
    #     args.config =  toml.load('config/ResiDNN.toml')
    #     # args.motif =  ["AAACA", "AAACT", "AGACC", "GAACA", "GAACT", "GGACC", "AAACC", "AGACA", "AGACT", "GAACC", "GGACA",
    #     #         "GGACT"]
    #     args.motif = "AAACA"
    #
    # if args.method == "MutliDNN":
    #     args.config =  toml.load('config/MutliDNN.toml')
    #
    # if args.method == "MutliDNN":
    args.config =  toml.load('config/GaussianAtt.toml')

def ObtainModel(args):

    if args.method == "ResiDNN":
        obj = ResiDNN(args)

    if args.method == "MutliDNN":
        obj = MutliDNN(args)


    return obj


#####################################
#
# DNN models
#
#####################################


class ResiDNN(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.args = args

        # build model
        ## site_init
        self.site_init1 = self.Init()
        self.site_init2 = self.Init()
        self.site_init3 = self.Init()
        self.site_init4 = self.Init()

        ## read_init
        self.read_init1 = self.Init()
        self.read_init2 = self.Init()
        self.read_init3 = self.Init()
        self.read_init4 = self.Init()

        ## inter_block
        self.inter_block1 = self.Inter()
        self.inter_block2 = self.Inter()
        self.inter_block3 = self.Inter()
        self.inter_block4 = self.Inter()
        self.inter_block5 = self.Inter()

        ## extract_block
        self.site_extract = self.Extract()
        self.block_extract = self.Extract()

        ## predict_block
        self.site_pre = self.site_pre_model()
        self.read_pre = self.read_pre_model()

    def build_layer(self, dims, pre=None, dropout=0):

        layers = []

        for index, item in enumerate(dims):
            input_channel = item[0]
            output_channel = item[1]
            layers.append(torch.nn.Linear(input_channel, output_channel))
            if not index==(len(dims)-1):
                layers.append(torch.nn.Dropout(p=dropout))
            if pre=="Sigmoid" and index==(len(dims)-1):
                layers.append(torch.nn.Sigmoid())
            elif pre=="IDENTITY" and index==(len(dims)-1):
                layers.append(torch.nn.Identity())
            else:
                layers.append(torch.nn.ReLU())

        return layers
    def Init(self):


        dims = self.args.config['block']['init']['dims']
        inte_model = nn.Sequential(*self.build_layer(dims, dropout=self.args.config['block']['init']['dropout']))

        return inte_model

    def Inter(self):


        dims = self.args.config['block']['inter']['dims']
        inter_model = nn.Sequential(*self.build_layer(dims, dropout=self.args.config['block']['inter']['dropout']))

        return inter_model

    def Extract(self):

        dims = self.args.config['block']['extract']['dims']
        read_model = nn.Sequential(*self.build_layer(dims, dropout=self.args.config['block']['extract']['dropout']))

        return read_model
    def site_pre_model(self):

        dims = self.args.config['block']['site_pre_model']['dims']
        activation = self.args.config['block']['site_pre_model']['activation']
        site_pre_model =  nn.Sequential(*self.build_layer(dims, pre=activation, dropout=self.args.config['block']['site_pre_model']['dropout']))

        return site_pre_model


    def read_pre_model(self):

        dims = self.args.config['block']['read_pre_model']['dims']
        activation = self.args.config['block']['read_pre_model']['activation']
        read_pre_model =  nn.Sequential(*self.build_layer(dims, pre=activation, dropout=self.args.config['block']['read_pre_model']['dropout']))

        return read_pre_model
    def forward(self, S, X):

        nums = X.size()[1]

        # Init Block
        ## ---> site init
        S0 = self.site_init1(S)
        feat_num = S0.size()[1]
        S0 = S0.repeat(1, nums)
        S0 = torch.reshape(S0,  (-1,nums, feat_num))

        S1 = self.site_init1(S)
        feat_num = S1.size()[1]
        S1 = S1.repeat(1, nums)
        S1 = torch.reshape(S1,  (-1,nums, feat_num))

        S2 = self.site_init1(S)
        feat_num = S2.size()[1]
        S2 = S2.repeat(1, nums)
        S2 = torch.reshape(S2,  (-1,nums, feat_num))

        S3 = self.site_init1(S)
        feat_num = S3.size()[1]
        S3 = S3.repeat(1, nums)
        S3 = torch.reshape(S3,  (-1,nums, feat_num))

        ## ---> read init
        R0 = self.read_init1(X)
        R1 = self.read_init1(X)
        R2 = self.read_init1(X)
        R3 = self.read_init1(X)

        # Residue Intermediate Block
        Rs1 = torch.concatenate([R1, S1], dim=2)
        Rs1_inter = self.inter_block1(Rs1)

        Rs2 = torch.concatenate([R2, S2], dim=2)
        Rs2_inter = self.inter_block2(Rs2)

        Rs3 = torch.concatenate([R3, S3], dim=2)
        Rs3_inter = self.inter_block3(Rs3)

        Rs4 = torch.concatenate([Rs1_inter, Rs2_inter], dim=2)
        Rs4_inter = self.inter_block4(Rs4)

        Rs5 = torch.concatenate([Rs4_inter, Rs3_inter], dim=2)
        Rs5_inter = self.inter_block5(Rs5)

        # information extract
        ## read information extract
        read_nn = torch.concatenate([Rs5_inter, R0], dim=2)
        # read_nn_bn =  self.apply_bn(read_nn)
        read_for = self.site_extract(read_nn)

        # read prob predict
        read_nn_bn = self.apply_bn(read_for)
        read_prob = self.read_pre(read_nn_bn)
        read_predict = torch.flatten(read_prob)


        ## site information extract
        site_nn = torch.concatenate([read_nn_bn, S0], dim=2)
        # site_nn_bn =  self.apply_bn(site_nn)
        site_for = self.site_extract(site_nn)

        # ratio predict
        # site_for_bn =  self.apply_bn(site_for)
        ratio_predict = self.site_pre(site_for)
        ratio_predict = torch.mean(ratio_predict, dim=1)

        return read_predict, ratio_predict

    def apply_bn(self, x):
        """ Batch normalization of 3D tensor x
        """
        bn_module = nn.BatchNorm1d(x.size()[1])
        device = (
            "cuda"
            if torch.cuda.is_available() and self.args.cuda
            else "cpu"
        )
        bn_module.to(device)
        return bn_module(x)

class MutliDNN(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.args = args

        # build model
        self.site_model = self.ExtractSite()
        self.read_model = self.ExtractRead()
        self.site_for = self.site_forward()
        self.site_pre = self.site_pre_model()
        self.read_pre = self.read_pre_model()

    def build_layer(self, dims, pre=None, dropout=0):

        layers = []

        for index, item in enumerate(dims):
            input_channel = item[0]
            output_channel = item[1]
            layers.append(torch.nn.Linear(input_channel, output_channel))
            if not index==(len(dims)-1):
                layers.append(torch.nn.Dropout(p=dropout))
            if pre=="Sigmoid" and index==(len(dims)-1):
                layers.append(torch.nn.Sigmoid())
            elif pre=="IDENTITY" and index==(len(dims)-1):
                layers.append(torch.nn.Identity())
            else:
                layers.append(torch.nn.ReLU())

        return layers
    def ExtractSite(self):


        dims = self.args.config['block']['ExtractSite']['dims']
        site_model = nn.Sequential(*self.build_layer(dims, dropout=self.args.config['block']['ExtractSite']['dropout']))

        return site_model

    def ExtractRead(self):

        dims = self.args.config['block']['ExtractRead']['dims']
        read_model = nn.Sequential(*self.build_layer(dims, dropout=self.args.config['block']['ExtractRead']['dropout']))

        return read_model
    def site_pre_model(self):

        dims = self.args.config['block']['site_pre_model']['dims']
        activation = self.args.config['block']['site_pre_model']['activation']
        site_pre_model =  nn.Sequential(*self.build_layer(dims, pre=activation, dropout=self.args.config['block']['site_pre_model']['dropout']))

        return site_pre_model

    def site_forward(self):

        dims = self.args.config['block']['site_forward']['dims']
        site_pre_model =  nn.Sequential(*self.build_layer(dims,  dropout=self.args.config['block']['site_forward']['dropout']))

        return site_pre_model

    def read_pre_model(self):

        dims = self.args.config['block']['read_pre_model']['dims']
        activation = self.args.config['block']['read_pre_model']['activation']
        read_pre_model =  nn.Sequential(*self.build_layer(dims, pre=activation, dropout=self.args.config['block']['read_pre_model']['dropout']))

        return read_pre_model
    def forward(self, S, X):

        nums = X.size()[1]
        # site information extract
        site_nn = self.site_model(S)

        ## site->read
        # site_nn_bn = self.apply_bn(site_nn)
        Sn_rep = site_nn.repeat(1, nums)
        Sn_rep = torch.reshape(Sn_rep,  (-1,nums, 20))
        siteToread = torch.concatenate([X, Sn_rep], dim=2)

        # read information extract
        read_nn = self.read_model(siteToread)

        ## read prob predict
        read_nn_nn_bn = self.apply_bn(read_nn)
        read_prob = self.read_pre(read_nn_nn_bn)
        read_prob = torch.flatten(read_prob)

        ## read->site
        S_rep = S.repeat(1, nums)
        S_rep = torch.reshape(S_rep,  (-1, nums, 20))
        readTosite = torch.concatenate([read_nn, S_rep], dim=2)

        ## site ratio predict
        # readTosite_bn = self.apply_bn(readTosite)
        ratio_forward = self.site_for(readTosite)
        ratio_forward_bn =  self.apply_bn(ratio_forward)
        ratio = self.site_pre(ratio_forward_bn)
        ratio = torch.mean(ratio, dim=1)
        return read_prob, ratio

    def apply_bn(self, x):
        """ Batch normalization of 3D tensor x
        """
        bn_module = nn.BatchNorm1d(x.size()[1])
        device = (
            "cuda"
            if torch.cuda.is_available() and self.args.cuda
            else "cpu"
        )
        bn_module.to(device)
        return bn_module(x)
