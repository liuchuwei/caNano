import os
import numpy as np
import torch

def predict_multidnn(model, motif, args, device, dataset):

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

                        # site information extract
                        site_nn = model.site_model(S)

                        # read information extract
                        read_nn = model.read_model(X)

                        # concate read and site information
                        ## read->site
                        S_rep = S.repeat(1, n_reads[idx])
                        S_rep = torch.reshape(S_rep, (-1, 20))
                        readTosite = torch.concatenate([read_nn, S_rep], dim=1)

                        ## site->read
                        Sn_rep = site_nn.repeat(1, n_reads[idx])
                        Sn_rep = torch.reshape(Sn_rep, (-1, 20))
                        siteToread = torch.concatenate([X, Sn_rep], dim=1)

                        ## Merge
                        meta = torch.concatenate([readTosite, siteToread], dim=1)

                        ## read predict
                        read_predict = model.read_pre(meta)

                        for ids, pro in zip(read_ids[idx], read_predict.cpu().numpy()):
                            r.write('%s, %.16f\n' % (ids, pro))

                        ## read_predict_site
                        n_iters = args.num_iterations
                        n_samples = args.num_samples
                        proba = np.random.choice(torch.squeeze(read_predict).cpu(), n_iters * n_samples,
                                                 replace=True).reshape(n_iters,
                                                                       n_samples)
                        # read_predict_site = (1 - np.prod(1 - proba, axis=1)).mean()
                        read_predict_site = (np.max(proba, axis=1)).mean()

                        s.write('%s, %.16f\n' % (site_ids[idx], read_predict_site))

    cmd = "rm %s/%s.tsv" % (args.output, tpp)
    os.system(cmd)
    print("finish!")