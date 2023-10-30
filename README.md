# caNano
## Dependence
![](https://img.shields.io/badge/software-version-blue)  
[![](https://img.shields.io/badge/Guppy-v6.5.7-green)](https://community.nanoporetech.com/downloads)
[![](https://img.shields.io/badge/Minimap2-v2.24-green)](https://github.com/lh3/minimap2)
[![](https://img.shields.io/badge/samtools-v1.1.7-green)](https://github.com/samtools/samtools)   
[![](https://img.shields.io/badge/bedtools-v2.29.1-blue)](https://bedtools.readthedocs.io/en/latest/)
[![](https://img.shields.io/badge/ELIGOS-v2.0.1-blue)](https://gitlab.com/piroonj/eligos2)
[![](https://img.shields.io/badge/Epinano-v1.2.0-blue)](https://github.com/novoalab/EpiNano)  
[![](https://img.shields.io/badge/MINES-v0.0-orange)](https://github.com/YeoLab/MINES.git)
[![](https://img.shields.io/badge/Tombo-v1.5.1-orange)](https://github.com/nanoporetech/tombo)
[![](https://img.shields.io/badge/Nanom6A-v2.0-orange)](https://github.com/gaoyubang/nanom6A)  
[![](https://img.shields.io/badge/m6Anet-v1.0-purple)](https://github.com/GoekeLab/m6anet) 
[![](https://img.shields.io/badge/nanopolish-v0.14.0-purple)](https://github.com/jts/nanopolish)  

## Genome
[![](https://img.shields.io/badge/mm39-orange)](https://hgdownload.soe.ucsc.edu/goldenPath/mm39/bigZips/)
[![](https://img.shields.io/badge/hg38-green)](https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/)


<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary><h2 style="display: inline-block">Table of Contents</h2></summary>
  <ol>
    <li>
      <a href="#Intallation">Intallation</a>
    </li>
    <li><a href="#Usage">Usage</a></li>
    <li><a href="#References">References</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#Contact">Contact</a></li>
  </ol>
</details>

## Intallation
1.Clone the project
   ```shell
   git clone https://github.com/liuchuwei/caNano.git
   ```
1.Install conda environment
   ```shell
   conda env create -f caNano.yaml
   ```
2.prepare tookit: 

check and modify the tookit.py file (in 'utils' directory).
    
## Usage
1.Basecalling
   ```shell
   python 01.basecalling.py -i $fast5 -o $out
   ```
2.Resguiggle

preprocess

   ```shell
   conda activate tombo
   python 02.resquiggle_pre.py -f $fast5 -o $out
   ```
annotate_raw_with_fastqs

   ```shell
   cat *.fastq > merge.fastq
   python 03.resquiggle.py preprocess annotate_raw_with_fastqs \
--fast5-basedir $single \
--fastq-filenames $merge_fastq \
--overwrite \
--processes 8
   ```
resquiggling
   ```shell
python 03.resquiggle.py resquiggle $fast5 $reference \
--rna \
--corrected-group RawGenomeCorrected_000 \
--basecall-group Basecall_1D_000 \
--overwrite \
--processes 16 \
--fit-global-scale \
--include-event-stdev
   ```
3.Minimap
   ```shell
python 04.minimap.py -i <directory of fastq files> -o <output directory> -r <path of reference>
   ```

4.Eventalign: preprocess data for m6anet model, you can skip this step if you don't use m6anet model.

```shell
python 05.eventalign.py -f <directory of fast5 files> -o <output directory> \
 -fq <path of fastq> -r <path of reference> -bam <path of bam files> -o <output directory>
```

5.m6a detection

(1) caNano

activate environment
   ```shell
   conda activate caNano
   ```

preprocess
   ```shell
   python caNano.py preprocess --single $single_fast5 -o $output -g $genome.fa -r $transcript.fa -i $gene2transcripts.txt -b $bam
   ```

train
   ```shell
   python caNano.py train --mod %mod.tsv --unmod %unmod.tsv --out %output
   ```

predict
   ```shell
   python caNano.py predict --input %feature.tsv --output %output --model %model
   ```

(2) Tombo

activate environment
```shell
conda activate tombo
```

predict
```shell
python tombo.py detect_modifications de_novo --fast5-basedirs <directory of fast5 files> \
--statistics-file-basename <output name> \
--corrected-group RawGenomeCorrected_000 \
--processes 16
```

output
```shell
python tombo.py text_output browser_files --fast5-basedirs <directory of fast5 files> \
--statistics-filename <output name>  \
--browser-file-basename wt_rrach \
--genome-fasta <path of reference> \
--motif-descriptions RRACH:3:m6A \
--file-types coverage dampened_fraction fraction \
--corrected-group RawGenomeCorrected_000
```

(3) Mines

activate environment
```shell
conda activate Mines
```

tidy tombo result
```shell
awk '{if($0!=null){print $0}}' wt.fraction_modified_reads.plus.wig > wt.wig
wig2bed < wt.wig > wt.fraction_modified_reads.plus.wig.bed --multisplit=mines
```

predict
```shell
python Mines.py --fraction_modified $tombo/wt.fraction_modified_reads.plus.wig.bed \
--coverage $tombo/wt.coverage.plus.bedgraph \
--output wt.bed \
--ref $ref \
--kmer_models $MINES/Final_Models/names.txt
```

(4) m6anet

data preprocess
```shell
m6anet dataprep --eventalign wt_eventalign.txt
--out_dir wt
--n_processes 16
--readcount_max 2000000
```

predict
```shell
m6anet inference --input_dir wt
--out_dir run/wt
--n_processes 16
```

(5) Nanom6A

list all fast5 files
```shell
find single -name "*.fast5" >files.txt
```

extracting signals
```shell
extract_raw_and_feature_fast --cpu=20 --fl=files.txt -o result --clip=10
```
predict
```shell
predict_sites --cpu 20 -i result -o result_final -r data/cc_ref.fa -g data/cc_ref.fa -b data/gene2transcripts.txt --model Nanom6A/model
```

(6) Eligos
```shell
python ELGOS rna_mod -i <bam file> -reg <bed files> -ref <REFERENCE> -m <rBEM5+2 model> -p <output file prefix> \
-o <output file directory> --sub_bam_dir <SUB_BAM_DIR> \
--max_depth 2000000 --min_depth 5 --esb 0 --oddR 1 --pval 1 -t 16
```

(7) Epinano

extract features
```shell
python 07.Epinano_Variants.py -R $ref -b <wt / ko bam file> -s <path to sam2tsv> -n 16 -T t
```

slide features
```shell
python 07.Epinano_slide_feature.py <per.site.csv> 5
```

preict
```shell
python 07.Epinano_Predict.py \
--model $Epinano/models/rrach.q3.mis3.del3.linear.dump \
--predict wt.plus_strand.per.site.5mer.csv \
--columns 8,13,23 \
--out_prefix wt
```

## License
Distributed under the GPL-2.0 License License. See LICENSE for more information.

## Contact
liuchw3@mail2.sysu.edu.cn

## Reference

