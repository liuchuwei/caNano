import argparse
from argparse import ArgumentDefaultsHelpFormatter
from utils.LoadData import DataLoad
from utils.train_utils import set_seed, train_model
from model.model_factory import ObtainModel, loadconfig

def argparser():
    parser = argparse.ArgumentParser(
        formatter_class=ArgumentDefaultsHelpFormatter,
        add_help=False
    )
    # load data
    parser.add_argument('--signal', required=True, help='Path of sample: signal input')
    parser.add_argument('--groundtruth', required=True, help='Path of groundtruth')
    parser.add_argument('--mod', required=False, help='Path of modified sample')
    # parser.add_argument('--unmod', required=False, help='Path of non modified sample')
    parser.add_argument('--motif', default="AAACA", help='Define the motif')
    # parser.add_argument('--ratio', default="1:1", help='Ratio of modified sample and non modified sample')
    # parser.add_argument('--min_reads', default="20", help='Minimum reads for site')

    # seed
    parser.add_argument('--seed', required=False, default=666, help='Path of non modified sample')

    # model
    parser.add_argument('--method', dest='method', default="GaussianAtt.toml",
                        help='Method. Possible values:RsiDNN; MutliDNN; GaussianAtt.toml')

    # device
    parser.add_argument('--cuda', dest='cuda', type=bool,default=True,
                        help='Whether to use gpu')

    # save model
    parser.add_argument('--out', required=True, help='Directory for saving models')


    return parser


def main(args):

    '1.set seed'
    set_seed(args.seed)

    '2.load config'
    loadconfig(args)

    '3.load data'
    train_dl, val_dl, test_dl = DataLoad(args).buildDataloder()

    '4.build model'
    model = ObtainModel(args)

    '5.train model'
    train_model(args, dataset=[train_dl, val_dl, test_dl] , model=model)
