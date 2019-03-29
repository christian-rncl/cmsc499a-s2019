
Paper: Multi-task learning for Host-Pathogen protein interactions
Thank you for your interest in this work.

Author: Meghana Kshirsagar (meghanak@ymail.com)
Files:
======
	* Code:       All code is in one file: src/pathopt/MultitaskPathOpt.java
	* Scripts:   'run.sh', 'compile.sh'
	* Data:       Sample datafiles for illustration in "data/" directory
	* Param-file: See below for a description of 'sample_[large/small]_datasplits.txt'.

Note:   
=====
(a) All provided data is sample data for illustration. It's a small fraction of the data used in the paper.

(b) The provided small scripts, commands assume Linux/Mac OS. To run on Windows, the paths will have to be modified accordingly. Change the "/" to "\" in 'run.sh', 'compile.sh' and 'sample_datasplits.txt'. The java code itself does not have any such issues.

(c) All data files follow the LibSVM data format. Each line of the data file represents one example/instance/datapoint. The format of each line is:
	<label> <featid>:<featval> <featid>:<featval> ...
This format allows a sparse representation of features. Missing features will simply not be listed.

(d) Format of pathway vectors file (see 'task1_pathway_vectors.txt', 'b_anthracis_pathway_vectors.txt'):
Each line in these files should contain the pathway vector of one positive example from the training data. The k-th line should correspond to the k-th positive example in the data. Each vector is an indicator vector of size 2100 (the number of human pathways as per Reactome+NCI pathway data).

(e) The code assumes that the input data (training/test/heldout) contains all positive examples first (label: 1) followed by all negative class examples (label: -1).


Compiling using provided script:
--------------------------------
sh compile.sh


Running with provided script:
-----------------------------

sh run.sh <data-splits-file> <regParam-lambda> <regParam-sigma> <class-cost-ratio> <file-out-weight>
(See the next section for an explanation of the parameters)

(a) With default parameter values on the small dummy dataset:
		sh run.sh sample_small_datasplits.txt

(b) With tasks, lambda=0.001, sigma=1, and positive:negative class-skew = 1:5
		sh run.sh sample_small_datasplits.txt 0.001 1 5

(c) Running on the larger dataset, with different parameters. The output is directed to files std.out std.err:
		sh run.sh sample_large_datasplits.txt 0.001 10 100  1>std.out  2>std.err


Java code Usage: 
----------------
java -Xmx6g -cp mallet.jar:mallet-deps.jar:bin pathopt.MultitaskPathOpt <data-splits-file> <regParam-lambda> <regParam-sigma> <class-cost-ratio> <file-out-weight>

The first parameter is mandatory. The rest are optional and will be assigned default values if missing.
<data-splits-file> : a file containing the location of the training, test and held-out folds of all tasks.
                     Please check the provided 'sample_[small/large]_datasplits.txt' for an example.
<regParam-lambda>  : parameter indicating importance of the pathway regularizer term (default: 0.01)
<regParam-sigma>   : parameter that controls importance of the L2 regularizer term (default: 1)
<class-cost-ratio> : fraction indicating the pos:neg class skew in data. See the end for an example.
                     The loss terms will be computed accordingly (default: 1)
<file-out-weight>  : location for an output file to save the learned weights (default: none)

Example for assigning <class-cost-ratio>: If the data has 100 times more negatives for a single positive, set this value to 100.


Output from the code:
----------------------
See sample output files std.out and std.err to get an idea.
The code outputs information while reading in the datasets, and then optimization progress and finally the classification performance (precision, recall, f1) on the training, held-out and test data for each task.
