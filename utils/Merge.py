# -*- coding: utf-8 -*-
import gzip

from collections import defaultdict
from tqdm import tqdm


def obtain_idsTiso(args):
    basefl = '/'.join(args.output.split("/")[:-1])
    fl = "%s/extract.reference.isoform.bed12" % (basefl)
    idsTiso = {}
    for i in open(fl, "r"):
        ele = i.rstrip().split()
        idsTiso[ele[3]] = ele[0]
    return idsTiso


def obtain_siteInfo(args):
    basefl = '/'.join(args.output.split("/")[:-1])
    fl = "%s/extract.sort.plus_strand.per.site.csv" % (basefl)
    siteInfo = defaultdict(dict)
    for i in open(fl, "r"):
        if i.startswith("#"):
            pre1 = "#"
            continue
        ele = i.rstrip().split(',')
        siteInfo[ele[0]][ele[1]] = ele[2:]

    return siteInfo


def obtain_genoInfo(args):
    basefl = '/'.join(args.output.split("/")[:-1])
    fl = "{0}/extract.reference.bed12".format(basefl)
    readgene = {}
    for i in open(fl, "r"):
        ele = i.rstrip().split()
        readgene[ele[3]] = [ele[0], ele[1], ele[2]]
    return readgene

def Merge_seq_current(fl, idsTiso, readgene, siteInfo):

    ## 4.merge information
    # fl = "".join([str(x[0]) for x in nums])
    # fl = fl.split("\n")
    # for i in fl[:-1]:
    # ele = i.rstrip().split()
    try:
        ele = fl.rstrip().split()
        ids = ele[0].split("|")[0]
        pos = ele[0].split("|")[1]
        isoform = idsTiso[ids]
        genemap = readgene[ids]
        site = str(int(pos) + int(genemap[1]) + 1)
    except:
        return None

    if isoform in siteInfo and site in siteInfo[isoform]:
        align_event = siteInfo[isoform][site]
        ele.append("|".join(align_event))  # base, strand, cov, q_mean, q_median, q_std, mis, ins, del
        ele.append("|".join(genemap))
        lines = "\t".join(ele) + "\n"
        return lines
    else:
        return None

def MergeSeqCurrent_single_motif(args, nums):
    'Merege sequence and current information'

    basefl = '/'.join(args.output.split("/")[:-1])
    storepos = defaultdict(dict)
    fl = "".join([str(x[0]) for x in nums])
    fl = fl.split("\n")
    for i in fl[:-1]:
        # ~ c9a0d84d-42f6-456c-895f-e957d1172623|6|AAGGGAAAGACTCCAGAGGAAATTAGGAAGACCTTTAACATCAAGAATGACTTTACACCTGAGGAGGAGGAGGAAGTTCGCCGTGAGAACCAGTGGGCATTTGAATGAAGTGCGTCTGATGGTTTCATGGAAGGAATGTTGTTCTAATGCCAAATGAATGCTGTGGGTTATCTTAGCGTAGACAAGACTATGTTTCTATGACTTTATTGTGAACCTGTGAGCACATTGACTGTAAATAATACTTGTATTCTGGGGAGGGGATTGGTAGTAGTTTCCTGCAATCAATCCTCTGCTTGTGGGCAAATGTTATTTGTTGCAGACTTGCAGTGATCCTTATCTGTTGTATCTGTTTTCCCTCTGTGTTCCTGCCAAGTTTGTTTCTTGGACATAATCATCAAGTCTTGGTGTCTCTT	1.0	0.06883740425109862	0.9311625957489014
        # ~ 0.09221864	0.90778136	GXB01149_20180715_FAH87828_GA10000_sequencing_run_20180715_NPL0183_I1_33361_read_17200_ch_182_strand.fast5|177|2,1,3,2,0,1,3,2,0,3,1,1,0,0,0,3,1,1,3,2,0,3,2,0,1,1,1,3,1,3,3,2,3,0,1,1,0,2,0,2,0,3,3,2,1,1,1,0,3,0,3,2,3,0,1,0,0,2,0,1,3,2,0,1,1,2,0,2,1,1,0,0,0,3,0,3,2,0,0,2,1,1,0,1,3,2,1,3,1,2,1,0,2,3,3,2,2,0,1,0,1,0,2,0,0,2,3,0,3,2,1,1,0,3,2,2,2,1,3,2,0,2,2,1,3,2,0,2,2,3,3,0,3,2,2,0,0,0,1,0,0,3,2,3,0,3,0,0,0,3,2,2,1,3,3,2,1,2,3,3,3,0,3,3,2,3,3,0,3,2,3,2,3,3,2,0,0,0,1,0,3,2,2,3,1,3,2,3,3,3,0,1,3,1,3,3,3,3,2,2,2,2,3,3,2,2,3,3,3,3,2,3,2,0,2,2,2,3,3,3,2,0,0,3,3,3,1,0,3,0,0,2,0,0,3,2,0,0,3,2,0,3,0,3,3,3,1,2,3,2,1,0,2,1,3,1,1,0,0,0,1,3,0,3,2,0,3,3,3,2,2,2,2,2,3,3,2,0,0,3,2,2,0,0,0,3,0
        ele = i.rstrip().split()
        mark = ele[0]
        sig = ele[1:]
        ids, spos, seq = mark.split("|")
        # ~ ids=namechange[ids]
        # ~ print(ids,int(spos))
        storepos[ids][int(spos)] = sig

    fl = args.output + ".feature.fa"
    store = {}
    lines = open(fl, "r").readlines()
    for index, i in enumerate(lines):
        if i.startswith(">"):
            ids = i.rstrip().lstrip(">")
            read = lines[index + 1].rstrip()
            store[ids] = read

    fl = "{0}/extract.reference.bed12".format(basefl)

    readgene = {}
    for i in open(fl, "r"):
        ele = i.rstrip().split()
        readgene[ele[3]] = ele[0]
    fl = "%s/extract.sort.bam.tsv.gz" % (basefl)
    ##########################################
    ##########################################
    # ~ chr04	W_003002_20180416_FAH83697_MN23410_sequencing_run_20180415_FAH83697_mRNA_WT_Col0_2918_23801_read_57_ch_290_strand.fast5	-	10201753	10201753	236|10203065|GAACA	275|10202917|AGACC	991|10202201|GGACA	1003|10202189|AGACC	1373|10201819|AAACA
    # ~ c1,c2,c3=0,0,0

    pbar = tqdm(total=len(store.keys()), position=0, leave=True)
    pre1, pre2 = "", ""
    results = []
    for i in gzip.open(fl, "r"):
        i = i.decode("utf-8").rstrip()

        # ~ NR_002323.2	0	chr22	7541	A	I	31375381	a	M
        if i.startswith("#"):
            pre1 = "#"
            continue
        ele = i.rstrip().split()

        if ele[3] == "." or ele[6] == ".":
            continue

        align = [0, 0, 0, 0, 0]  # mat,mis,ins,del,qual
        if ele[-1] in ['M', 'm']:
            align[4] = ord(ele[-4]) - 33
            if (ele[-2] != ele[4]):
                align[1] = 1
            else:
                align[0] = 1

        if ele[-1] == 'D':
            align[3] = 1

        if ele[-1] == 'I':
            align[2] = 1

        ids, chro, idspos, gpos = ele[0], ele[2], int(ele[3]), ele[6]

        if ids != pre1:
            pbar.update(1)
            pre1 = ids

        if ele[1] == "0":
            strand = "+"

        elif ele[1] == "16":
            strand = "-"
            lens = len(store[ids])
            idspos = lens - idspos - 1

        if ids in storepos and idspos in storepos[ids] and ids in readgene:
            # kmer = store[ids][idspos - 2:idspos + 3]
            # line = "%s|%s|%s" % (idspos, gpos, kmer)
            # total_m6A_reads["%s\t%s\t%s\t%s\tNA\t" % (chro, ids, strand, readgene[ids])][line] = 1

            # ids|chro|strand|idspos|gpos|gene, base|kmer, mean, std, md_intense, length, align
            lines = "%s|%s|%s|%s|%s|%s\t%s|%s\t%s\t%s\t%s\t%s\t%s\n" % (ids, chro, strand, idspos, gpos, readgene[ids],
                                                                        storepos[ids][idspos][0],
                                                                        storepos[ids][idspos][1],
                                                                        storepos[ids][idspos][2],
                                                                        storepos[ids][idspos][3],
                                                                        storepos[ids][idspos][4],
                                                                        storepos[ids][idspos][5],
                                                                        "|".join(str(item) for item in align))
            results.append(lines)

    return results