#!/usr/bin/env Python
# coding=utf-8
import argparse
import multiprocessing
import os
from argparse import ArgumentDefaultsHelpFormatter
from utils.ExtractSeqCurrent import extract_feature
from utils.Mapping import mapping
from utils.Merge import Merge_seq_current, obtain_idsTiso, obtain_siteInfo, obtain_genoInfo
from utils.AlignVariant import align_variant
from tqdm import tqdm

def argparser():
    parser = argparse.ArgumentParser(
        formatter_class=ArgumentDefaultsHelpFormatter,
        add_help=False
    )
    parser.add_argument('--single', required=True, help='Single fast5 path')
    parser.add_argument('--kmer', default='5', help='Length of kmer')
    parser.add_argument('--kmer_filter', default='[AG][AG]AC[ACT]', help='Define kmer filter')
    parser.add_argument('--basecall_group', default="RawGenomeCorrected_000",
                        help='The attribute group to extract the training data from. e.g. RawGenomeCorrected_000')
    parser.add_argument('--basecall_subgroup', default='BaseCalled_template',
                        help='Basecall subgroup Nanoraw resquiggle into. Default is BaseCalled_template')
    parser.add_argument('--clip', default=10, help='Reads first and last N base signal drop out')
    parser.add_argument('-o', '--output', required=True, help="Output directory")
    parser.add_argument('-g', '--genome', required=True, help="Genome file for mapping")
    parser.add_argument('-r', '--reference', required=True, help="Referance transcripts sequence file")
    parser.add_argument('-b', '--isoform', required=True, help="Gene to referance transcripts information")
    parser.add_argument('--cpu', default=8, help='cpu number usage,default=8')
    parser.add_argument('--support', default=10,
                        help='The minimum number of DRS reads supporting a modified m6A site in genomic coordinates from one million DRS reads.  The default is 10.  Due to the low sequencing depth for DRS reads, quantification of m6A modification in low abundance gene is difficult.  With this option, the pipeline will attempt to normalize library using this formula: Total number of DRS reads/1,000, 000 to generate \'per million scaling factor\'.   Then the  \'per million scaling factor\'  multiply reads from -r option to generate the cuttoff for the number of modified transcripts  for each modified m6A site.   For example, the option (-r = 10, total DRS reads=2, 000, 000) will generate (2000000/1000000)*10=20 as cuttoff. Than means that modified A base supported by at least 20 modified transcripts will be identified as modified m6A sites in genomic coordinates.')

    return parser


def main(args):

    'main funtion for preprocess'
    '1.Get path of single fast5 files'
    print("Get path of single fast5 files...")
    fls =  [args.single + "/" + item for item in os.listdir(args.single)]

    '2.Extract seq & current information'
    print("Extract seq & current information...")
    pool = multiprocessing.Pool(processes = int(args.cpu))

    results=[]
    for fl in fls:
        result=pool.apply_async(extract_feature,(fl,args))
        results.append(result)
    pool.close()

    pbar = tqdm(total=len(fls), position=0, leave=True)
    nums = []
    for result in results:
        num, seq = result.get()
        if num and seq:
            nums.append([num, seq])
        pbar.update(1)
    pool.join()

    dirs = args.output.split("/")
    dirs_list = []
    for i in range(len(dirs)):
        dirs_list.append("/".join(dirs[0:i + 1]))

    for item in dirs_list[:-1]:
        if not os.path.exists(item):
            os.mkdir(item)

    output = open(args.output + ".feature.fa", "w")
    output.write("".join([str(x[1]) for x in nums]))
    output.close()

    '3.Mapping with genome'
    print("Mapping...")
    mapping(args, nums)

    '4.Extract align variant information'
    print("Extract align variant information...")
    align_variant(args)

    '5.Merge RRACH seq & current information'
    print("Merge RRACH seq & current information...")
    ## 1.ids to isoform
    idsTiso = obtain_idsTiso(args)

    ## 2.site information
    siteInfo = obtain_siteInfo(args)

    ## 3.geno information
    readgene = obtain_genoInfo(args)
    fls = "".join([str(x[0]) for x in nums])
    fls = fls.split("\n")
    pool = multiprocessing.Pool(processes = int(args.cpu))

    results = []
    for fl in fls[:-1]:
        result = pool.apply_async(Merge_seq_current, (fl, idsTiso, readgene, siteInfo))
        results.append(result)
    pool.close()

    pbar = tqdm(total=len(fls)-1, position=0, leave=True)
    meta = []
    for result in results:
        lines = result.get()
        if lines:
            meta.append(lines)
        pbar.update(1)
    pool.join()

    output = open(args.output + ".feature.tsv", "w")
    output.write("".join([x for x in meta]))
    output.close()