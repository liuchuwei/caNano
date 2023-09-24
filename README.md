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
   ```sh
   python 02.resquiggle.py -i $fastq -o $out -r $fasta
   ```
3.Run caNano for m6a detection

preprocess
   ```sh
   git clone https://github.com/liuchuwei/PGLCN.git
   ```

## License
Distributed under the GPL-2.0 License License. See LICENSE for more information.

## Contact
liuchw3@mail2.sysu.edu.cn

## Reference

