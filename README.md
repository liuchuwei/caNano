# caNano
## Dependence
![](https://img.shields.io/badge/software-version-blue)  
[![](https://img.shields.io/badge/Guppy-v6.5.7-green)](https://community.nanoporetech.com/downloads)
[![](https://img.shields.io/badge/Minimap2-v2.24-green)](https://github.com/lh3/minimap2)
[![](https://img.shields.io/badge/samtools-v1.1.7-green)](https://github.com/samtools/samtools)  


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
   ```sh
   git clone https://github.com/liuchuwei/caNano.git
   ```
1.Install conda environment
   ```sh
   conda env create -f caNano.yaml
   ```
2.prepare tookit: 

check and modify the tookit.py file (in 'utils' directory).
    
## Usage
1.Basecalling
   ```sh
   python 01.basecalling.py -i $fast5 -o $out
   ```
2.Resguiggle

preprocess

   ```sh
   conda activate tombo
   python 02.resquiggle_pre.py -f $fast5 -o $out
   ```
annotate_raw_with_fastqs

   ```sh
   cat *.fastq > merge_fastq
   python 03.resquiggle.py preprocess annotate_raw_with_fastqs \
--fast5-basedir $single \
--fastq-filenames $merge_fastq \
--overwrite \
--processes 8
   ```
resquiggling
   ```sh
python 03.resquiggle.py resquiggle $fast5 $reference \
--rna \
--corrected-group RawGenomeCorrected_000 \
--basecall-group Basecall_1D_000 \
--overwrite \
--processes 16 \
--fit-global-scale \
--include-event-stdev
   ```
3.Run caNano for m6a detection

activate environment
   ```sh
   conda activate caNano
   ```

preprocess
   ```sh
   python caNano.py preprocess --single $single_fast5 -o $output -g $genome.fa -r $transcript.fa -b $gene2transcripts.txt
   ```
## License
Distributed under the GPL-2.0 License License. See LICENSE for more information.

## Contact
liuchw3@mail2.sysu.edu.cn

## Reference

