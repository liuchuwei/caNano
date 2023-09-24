# -*- coding: utf-8 -*-

import argparse
import sys
import os

from utils.tookit import Tookits

# args define
parser = argparse.ArgumentParser(description='Basecalling')
parser.add_argument('-f', '--fast5', required=True, help="directory of fast5 files")
parser.add_argument('-fq', '--fastq', required=True, help="directory of merge fastq files")
parser.add_argument('-r', '--reference', required=True, help="directory of reference files")
parser.add_argument('-o', '--output', required=True, help="output directory")
parser.add_argument('-t', '--process', default=40, help="number of thread")
args = parser.parse_args(sys.argv[1:])
global FLAGS
FLAGS = args

tools = Tookits()

# # Convert merged single big fast5 into small size fast5 file
fl = os.listdir(FLAGS.fast5)[0]
fsize = os.path.getsize(os.path.join(FLAGS.fast5,fl))
fsize = fsize/float(1024*1024)

# create output directory
single_out = FLAGS.output + "/single"

dirs = single_out.split("/")
dirs_list = []
for i in range(len(dirs)):
    dirs_list.append("/".join(dirs[0:i + 1]))

for item in dirs_list:
    if not os.path.exists(item):
        os.mkdir(item)


if fsize > 10:

    cmd = "%s -i %s -s %s -t %s --recursive" % (tools.multi_to_single_fast5, FLAGS.fast5, single_out,FLAGS.process)
    os.system(cmd)

    FLAGS.fast5 = single_out

    # annotate_raw_with_fastqs
    cmd = "%s preprocess annotate_raw_with_fastqs \
    --fast5-basedir %s \
    --fastq-filenames %s \
    --overwrite \
    --processes %s" % (tools.tombo, FLAGS.fast5, FLAGS.fastq,FLAGS.process)
    os.system(cmd)

else:
    single_out = FLAGS.output + "/single"
    # cmd = "cp %s/*.fast5 %s" % (FLAGS.fast5, single_out)
    cmd = "find %s -name  '*.fast5' | xargs -i cp {} %s" % (FLAGS.fast5, single_out)
    os.system(cmd)
    FLAGS.fast5 = single_out

    # annotate_raw_with_fastqs
    cmd = "%s preprocess annotate_raw_with_fastqs \
    --fast5-basedir %s \
    --fastq-filenames %s \
    --overwrite \
    --processes %s" % (tools.tombo, FLAGS.fast5, FLAGS.fastq, FLAGS.process)
    os.system(cmd)

# resquiggle raw signals
cmd = "%s resquiggle %s %s \
--rna \
--corrected-group RawGenomeCorrected_000 \
--basecall-group Basecall_1D_000 \
--overwrite \
--processes %s \
--fit-global-scale \
--include-event-stdev" % (tools.tombo, FLAGS.fast5, FLAGS.reference, FLAGS.process)
os.system(cmd)
