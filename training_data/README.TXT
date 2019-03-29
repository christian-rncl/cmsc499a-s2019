
Author: Meghana Kshirsagar (mkshirsa@cs.cmu.edu)

Format of PPI files:
====================
<pathogen-UniprotID>  <pathogen-gene>  <host-UniprotID>  <host-gene>  <class>

The <class> = 0 for the negative protein pairs, and <class> = 1 for positive PPIs.

Format of Pathway vector files:
===============================
For each line in the positive-PPIs file, there is one line in the PathwayVectors file.
Ex: Bacillus_Anthracis_PositivePPIs.txt has 655 PPIs, so Bacillus_Anthracis_PositivePPIs_PathwayVectors.txt has 655 vectors.

Each vector is an indicator vector representing the set of human pathways corresponding to the human gene in the PPI.
Ex: The first line in Bacillus_Anthracis_PositivePPIs.txt is: 
Q81RG7 BA_2080 A6H8Y1 BDP1 1

The human gene is = BDP1
The pathway vector for BDP1 is the first line in Bacillus_Anthracis_PositivePPIs_PathwayVectors.txt 
There are 2100 pathways in the data we obtained from Reactome+NCBI. Thus the vector has 2100 columns. A column will have "1" if that pathway contains BDP1.

To know which pathway each column represents, please see the file : PathwayId_PathwayName.txt

The human-gene pathways mapping is present in the accompanying file: HumanGene_Pathway.map

Sources of data:
================
The positive PPIs files were obtained by downloading the host-pathogen PPIs from the PHISTO database and mapping the protein ids to the UniprotKB ids.
The negative pairs were obtained by randomly sampling the set of all possible host-pathogen protein pairs. The ratio of positive : negative pairs in the training data was roughly 1:100
The pathway vectors were obtained by using the "human-gene to pathway mapping" from Reactome database.
