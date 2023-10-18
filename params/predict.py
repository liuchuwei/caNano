import argparse
from argparse import ArgumentDefaultsHelpFormatter
from model.model_factory import loadconfig
from utils.LoadData import DataLoad
import os
import torch
from predict.predict_MutliDNN import predict_multidnn
from predict.predict_ResiDNN import predict_residnn

def argparser():
    parser = argparse.ArgumentParser(
        formatter_class=ArgumentDefaultsHelpFormatter,
        add_help=False
    )

    # load data
    parser.add_argument('--input', required=True, help='Path of sample after preprocess')
    parser.add_argument('--output', required=True, help='Path of output directory')
    parser.add_argument('--model', required=True, help='Path of model')

    # model
    parser.add_argument('--method', dest='method', default="ResiDNN",
                        help='Method. Possible values:ResiDNN, MutliDNN')

    # device
    parser.add_argument('--cuda', dest='cuda', type=bool,default=True,
                        help='Whether to use gpu')

    # predict
    parser.add_argument("--min_reads",
                        help='number of min reads.',
                        default=5, type=int)

    return parser


def main(args):

    '1.load config'
    loadconfig(args)

    '2.predict'
    print("Start predict")
    for tpp in ["AAACA", "AAACT", "AGACC", "GAACA", "GAACT", "GGACC", "AAACC", "AGACA", "AGACT", "GAACC", "GGACA",
                "GGACT"]:
        print("extract %s" % (tpp))
        cmd = "grep %s %s >%s/%s.tsv" % (tpp, args.input, args.output, tpp)
        os.system(cmd)
        args.motif = tpp
        args.fls = "%s/%s.tsv" % (args.output, tpp)
        dl = DataLoad(args, mod="Predict").buildDataloder()

        print("start predict %s" % (tpp))
        model = torch.load("%s/%s.mod"% (args.model, tpp))

        # device
        device = (
            "cuda"
            if torch.cuda.is_available() and args.cuda
            else "cpu"
        )

        model = model.to(device)

        if args.method == "MutliDNN":
            predict_multidnn(model=model, motif=tpp, args=args, device=device, dataset=dl)

        if args.method == "ResiDNN":
            predict_residnn(model=model, motif=tpp, args=args, device=device, dataset=dl)