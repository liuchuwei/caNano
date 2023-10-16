import argparse
import os
from argparse import ArgumentDefaultsHelpFormatter

import numpy as np

from sklearn.metrics import roc_auc_score

def argparser():
    parser = argparse.ArgumentParser(
        formatter_class=ArgumentDefaultsHelpFormatter,
        add_help=False
    )

    # load data
    parser.add_argument('--input', required=True, help='Path of predict directory')
    parser.add_argument('--output', required=True, help='Path of output directory')
    parser.add_argument('--groundtruth', required=True, help='Path of model')

    return parser

def main(args):

    '1.load preidct files'
    cmd = "cat %s/*.site_proba.csv > %s/total_predict.csv" % (args.input, args.output)
    os.system(cmd)

    fl = "%s/total_predict.csv" % (args.output)
    predict = []
    for i in open(fl, "r"):
        ele = i.rstrip().split(",")
        pro = float(ele[1])
        ele = ele[0].split("|")
        Chr, Start, End, Strand, Motif = ele[0], ele[1], ele[1], ele[2], ele[3]
        predict.append(["|".join([Chr, Start, End, Strand, Motif]), pro])


    '2.load groundtruth files'
    fl = args.groundtruth
    ground_truth = []
    for i in open(fl, "r"):

        if i.startswith("#"):
            continue

        ele = i.rstrip().split()

        motif = ele[4]

        if motif in ["AAACA", "AAACT", "AGACC", "GAACA", "GAACT", "GGACC", "AAACC", "AGACA", "AGACT", "GAACC", "GGACA",
                "GGACT"]:

            ground_truth.append("|".join(ele))

    '3.evaluate'
    total = list(set(ground_truth) & set([item[0] for item in predict]))

    y = np.array([item[0] in total for item in predict])
    y = y.astype(int)

    y_pred = np.array([item[1] for item in predict])
    roc_auc_score(y, y_pred)