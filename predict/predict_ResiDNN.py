import os
import random

import numpy as np
import torch


def predict_residnn(model, motif, args, device, dataset):
    tpp = motif
    dl = dataset

    with open(os.path.join(args.output, "%s.site_proba.csv" % (tpp)), 'a', encoding='utf-8') as s:
        with open(os.path.join(args.output, "%s.indiv_proba.csv" % (tpp)), 'a', encoding='utf-8') as r:
            with torch.no_grad():
                for it, data in enumerate(dl):
                    site_feature, features, n_reads, site_ids, read_ids = data

                    features = torch.split(features, tuple(n_reads.numpy()))
                    idx = [np.sum(n_reads.numpy()[:it]) for it in range(1, len(n_reads.numpy()) + 1)]
                    read_ids = np.split(read_ids, idx)

                    for idx, item in enumerate(features):


                        S = site_feature[idx].to(device)
                        X = item.to(device)

                        nums = X.size()[0]

                        # Init Block
                        ## ---> site init
                        S0 = model.site_init1(S)
                        feat_num = S0.size()[0]
                        S0 = S0.repeat(1, nums)
                        S0 = torch.reshape(S0, (-1, feat_num))

                        S1 = model.site_init1(S)
                        feat_num = S1.size()[0]
                        S1 = S1.repeat(1, nums)
                        S1 = torch.reshape(S1, (-1, feat_num))

                        S2 = model.site_init1(S)
                        feat_num = S2.size()[0]
                        S2 = S2.repeat(1, nums)
                        S2 = torch.reshape(S2, (-1, feat_num))

                        S3 = model.site_init1(S)
                        feat_num = S3.size()[0]
                        S3 = S3.repeat(1, nums)
                        S3 = torch.reshape(S3, (-1, feat_num))

                        ## ---> read init
                        R0 = model.read_init1(X)
                        R1 = model.read_init1(X)
                        R2 = model.read_init1(X)
                        R3 = model.read_init1(X)

                        # Residue Intermediate Block
                        Rs1 = torch.concatenate([R1, S1], dim=1)
                        Rs1_inter = model.inter_block1(Rs1)

                        Rs2 = torch.concatenate([R2, S2], dim=1)
                        Rs2_inter = model.inter_block2(Rs2)

                        Rs3 = torch.concatenate([R3, S3], dim=1)
                        Rs3_inter = model.inter_block3(Rs3)

                        Rs4 = torch.concatenate([Rs1_inter, Rs2_inter], dim=1)
                        Rs4_inter = model.inter_block4(Rs4)

                        Rs5 = torch.concatenate([Rs4_inter, Rs3_inter], dim=1)
                        Rs5_inter = model.inter_block5(Rs5)

                        # information extract
                        ## site information extract
                        site_nn = torch.concatenate([Rs5_inter, S0], dim=1)
                        site_for = model.site_extract(site_nn)

                        ## read information extract
                        read_nn = torch.concatenate([Rs5_inter, R0], dim=1)
                        read_for = model.site_extract(read_nn)

                        # read prob predict
                        read_nn_bn = model.apply_bn(read_for)
                        read_prob = model.read_pre(read_nn_bn)
                        read_prob = torch.flatten(read_prob)

                        for ids, pro in zip(read_ids[idx], read_prob.cpu().numpy()):
                            r.write('%s, %.16f\n' % (ids, pro))

                        # ratio predict
                        # site_for_bn = model.apply_bn(site_for)
                        ratio = model.site_pre(site_for)
                        ratio = torch.mean(ratio, dim=0)

                        s.write('%s, %.16f\n' % (site_ids[idx], ratio))

    cmd = "rm %s/%s.tsv" % (args.output, tpp)
    os.system(cmd)
    print("finish!")