import argparse
from argparse import ArgumentDefaultsHelpFormatter
from utils.LoadData import DataLoad
from utils.train_utils import set_seed
from model.model_factory import ObtainModel, loadconfig

def argparser():
    parser = argparse.ArgumentParser(
        formatter_class=ArgumentDefaultsHelpFormatter,
        add_help=False
    )
    # load data
    parser.add_argument('--mod', required=True, help='Path of modified sample')
    parser.add_argument('--unmod', required=True, help='Path of non modified sample')
    parser.add_argument('--motif', default="AAACA", help='Define the motif')
    parser.add_argument('--ratio', default="1:1", help='Ratio of modified sample and non modified sample')

    # seed
    parser.add_argument('--seed', required=False, default=666, help='Path of non modified sample')

    # model
    parser.add_argument('--method', dest='method', default="SingleDNN",
                        help='Method. Possible values:SingleDNN')

    return parser


def main(args):

    '1.set seed'
    set_seed(args.seed)

    '2.load config'
    loadconfig(args)

    '3.load data'
    dat = DataLoad(args)

    '4.build model'
    model = ObtainModel(args)

    '5.train model'

    '6.evaluate'