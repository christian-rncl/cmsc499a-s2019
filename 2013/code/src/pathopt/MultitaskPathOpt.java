package pathopt;


import cc.mallet.optimize.*;

import java.io.*;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.StringTokenizer;
import java.util.Arrays;

public class MultitaskPathOpt {

	private static final int NUM_PATHWAYS = 2100;

	private static final int LBFGS_HISTORY = 12;

	private static final double LBFGS_TOLERANCE = 0.0001;

	private static final int OPT_MAX_ITER = 80;
	
	private static final double POS_LABEL = 1;
	private static final double NEG_LABEL = -1;

	private static final double C = 10;  /** constant used in the function phi(z) = e^(z/C) **/

	/** weight vector related variables **/
	double[] combined_W; // combined weight vector of all tasks
	int[] task_W_startIdx; // index indicating the start offset for each task in the "combined_W" vector
	int[] task_W_endIdx; // index indicating the end offset for each task in the "combined_W" vector -- redundant info!
	double[] lambdas;	// stores user input regularizer parameters. Can use different lambda for different regularizer terms.
	/*-----------------*/

	/** data storing variables **/
	// for each task, for each example, this table stores the feature-id of its' features
	HashMap<Integer, ArrayList<ArrayList<Integer>>> task_FeatsStore; 	// the last element of each feature-ids list corresponds to the class label	

	// for each task, for each example, this table stores the values taken by its' features -- this datastructure is "parallel" to task_FeatsStore
	HashMap<Integer, ArrayList<ArrayList<Double>>> task_FeatValsStore;	// the last element of each values list stores the class (pos or neg)

	HashMap<Integer, double[]> task_predValsStore; // in each iteration of optimization, this stores w.x for each data instance of each task
	HashMap<Integer, int[][]> task_pathwayVectors; // for each task, stores pathway vectors in an array of size : nPos x N_pathways [the array size is n+ x N as per the paper notation]
	/*-----------------*/

	/** variables to store sizes/counts for each task **/
	ArrayList<Integer> task_trainSizes, task_numFeats, task_numPos, task_uniqFeats; // each of these lists has size = num_tasks
	int numParams, numPathways, numTasks, totalTrainSize;
	int maxNumFeats;

	/** optimization related variables **/
	double[] gradG, gradL; // gradients of the function G and of L_approx
	double regularizer_curr, G_prev, logLoss_curr, F_curr, L_approx, globalApprox, currValTrue=Double.MAX_VALUE, prevValTrue;
	double[] w_prev;

	/** variables to store intermediate results **/
	HashMap<Integer, double[]> pathwayFunc;	// stores f_k (first task) and g_k (second task) from Eqn(5) of paper.. one array per task, size of array = |numpathways|
	double costRatio;

	public MultitaskPathOpt(int nD, double[] larray, String[] trainFiles, String[] pathwayFiles, String[] hoFiles, String[] testFiles, double ratio) {
		System.out.println("Number of tasks: "+nD);

		// initialize all variables
		numPathways = NUM_PATHWAYS;
		costRatio = ratio;
		numTasks = nD;
		lambdas = larray;
		task_trainSizes = new ArrayList<Integer>(numTasks);
		task_numFeats = new ArrayList<Integer>(numTasks);
		task_uniqFeats = new ArrayList<Integer>(numTasks);
		task_numPos = new ArrayList<Integer>(numTasks);
		task_FeatsStore = new HashMap<Integer, ArrayList<ArrayList<Integer>>>(3*numTasks);
		task_FeatValsStore = new HashMap<Integer, ArrayList<ArrayList<Double>>>(3*numTasks);
		task_W_startIdx = new int[numTasks];
		task_W_endIdx = new int[numTasks];
		task_predValsStore = new HashMap<Integer, double[]>(numTasks);
		task_pathwayVectors = new HashMap<Integer, int[][]>(numTasks);

		// initialize intermediate results for each task
		pathwayFunc = new HashMap<Integer, double[]>(numTasks);
		for(int d=0; d<numTasks; d++) {
			double[] arr = new double[numPathways];
			pathwayFunc.put(d, arr);
		}

		// read in training data and pathway data of each task
		long startTime = System.currentTimeMillis();
		System.out.println("Reading Training data...");
		//totalTrainSize=0;
		for(int d=0; d<numTasks; d++) {
			readData(trainFiles[d], d);
			//totalTrainSize += trainSizes.get(d);
		}
		System.out.println("Reading Held-out data...");
		for(int d=0; d<numTasks; d++) {
			readData(hoFiles[d], numTasks+d);
		}
		System.out.println("Reading Test data...");
		for(int d=0; d<numTasks; d++) {
			readData(testFiles[d], 2*numTasks+d);
		}
		System.out.println("[READ:] Time taken: "+(System.currentTimeMillis()-startTime));
		System.out.println("Reading Pathways for positive examples...");
		for(int d=0; d<numTasks; d++) {
			readPathways(pathwayFiles[d], d);
		}

		// compute num of unique features
		maxNumFeats=0;
		for(int d=0; d<numTasks; d++) {
			int maxFeatNum;
			maxFeatNum = (task_numFeats.get(d) > task_numFeats.get(d+numTasks)) ? task_numFeats.get(d) : task_numFeats.get(d+numTasks);
			maxFeatNum = (task_numFeats.get(2*numTasks+d) > maxFeatNum) ? task_numFeats.get(2*numTasks+d) : maxFeatNum;
			task_uniqFeats.add(d, maxFeatNum);
			System.out.println("#Unique feats in task:"+d+" are:"+maxFeatNum);
			if(maxFeatNum > maxNumFeats)
				maxNumFeats = maxFeatNum;
		}

		System.out.println("Setting class cost to:"+costRatio);
		System.out.println("Max number of feats over all domains:"+maxNumFeats);
		System.out.println("Lambda: "+lambdas[0]+" Sigma: "+lambdas[1]);

		// initialize weight vectors
		initializeWeights();

		// initialize gradients
		gradG = new double[numParams];
		gradL = new double[numParams];
	}

	// inner optimization from Step-5 of algorithm in paper
	// input: numIters - maximum number of iterations allowed before convergence
	public void optimizeCvxApprox(int numIters) {

		// compute f_k and g_k (intermediate computation)
		computePathwayFunc(); 
		System.arraycopy(combined_W, 0, w_prev, 0, combined_W.length);

		getGradG(); // will be constant throughout the below inner optimization

		LBFGSOptimizer optimizable = new LBFGSOptimizer();
		Optimizer optimizer = new LimitedMemoryBFGS(optimizable);
		((LimitedMemoryBFGS)optimizer).setTolerance(LBFGS_TOLERANCE);
		//((LimitedMemoryBFGS)optimizer).setHistory(LBFGS_HISTORY);

		try {
			optimizer.optimize (numIters);
		} catch (InvalidOptimizableException e) {
			e.printStackTrace();
			System.err.println("Catching InvalidOptimizatinException! Saying converged!");
		} catch (OptimizationException e) {
			System.err.println("OptimizationException!!! Saying converged!");
			e.printStackTrace();
		} catch (Exception e) {
			System.err.println("Exception!!!");
			e.printStackTrace();
		}

		L_approx = -optimizable.funcVal;
	}

	// compute value of L
	public double computeTrueValue() {
		// debug: print the regularizer
		//printRegularizerVal();
		double G_curr = getvalG();
		globalApprox = L_approx - G_prev - dotProd(w_prev, gradG);
		prevValTrue = currValTrue;
		currValTrue = F_curr - G_curr;
		G_prev = G_curr;
		//System.out.println("APPROX VALUE: "+globalApprox+" TRUE VALUE: "+currValTrue+" DIFF: "+(currValTrue-globalApprox)); // debug
		return globalApprox;
	}

	// computes intermediate results and stores, for efficiency
	// computes f_i and g_i from Eqn (5)
	public void computePathwayFunc() {
		double maxPredVal=0, minPredVal=100;
		for(int d=0; d<numTasks; d++) {
			ArrayList<ArrayList<Integer>> trainFeats = task_FeatsStore.get(d); // feat ids for each example
			ArrayList<ArrayList<Double>> trainVals = task_FeatValsStore.get(d);
			int[][] pArray = task_pathwayVectors.get(d); // dimension: nPos x N

			// compute predicted values on all data instances for this task
			double[] predVals;
			if(task_predValsStore.containsKey(d))
				predVals = task_predValsStore.get(d);
			else
				predVals = new double[task_trainSizes.get(d)]; 

			for(int n=0; n<task_trainSizes.get(d); n++) {
				predVals[n] = dotProdwithW(trainFeats.get(n), trainVals.get(n), task_W_startIdx[d]);
				if(n < task_numPos.get(d)) {
					if(predVals[n] > maxPredVal)
						maxPredVal = predVals[n];
					else if(predVals[n] < minPredVal)
						minPredVal = predVals[n];
				}
			}
			task_predValsStore.put(d, predVals);

			// compute f_i or g_i below
			double[] arr = pathwayFunc.get(d);
			for(int k=0; k<numPathways; k++) {
				double sum=0, expLoss;
				for(int n=0; n<task_numPos.get(d); n++) {
					expLoss = Math.exp(predVals[n]/C);
					sum += pArray[n][k] * expLoss;
				}
				arr[k] = sum/((double)task_numPos.get(d));
			}
			pathwayFunc.put(d, arr);
		}
		//System.out.println("\t\t"+"[PathFunc:] MaxPredVal: "+maxPredVal+" MinPredVal: "+minPredVal);
	}

	// creates weight arrays and computes the start and end index
	public void initializeWeights() {
		int totFeats=0;
		for(int d=0; d<numTasks; d++) { // count number of features across train, holdout and test data
			task_W_startIdx[d] = totFeats;
			totFeats += task_uniqFeats.get(d);
			task_W_endIdx[d] = totFeats;
			System.out.println("W-limits: for task"+d+" Start:"+task_W_startIdx[d]+" End:"+task_W_endIdx[d]);
		}
		combined_W = new double[totFeats];
		w_prev = new double[totFeats];
		numParams = totFeats;
		System.out.println("Total number of features:"+totFeats);
	}

	public void printWeights(String filePrefix) {
		try{
			for(int d=0; d<numTasks; d++) {
				String fileName = filePrefix+"-w_"+d+".txt";
				System.out.println("Writing weights to file "+fileName);
				BufferedWriter out = new BufferedWriter(new FileWriter(fileName));
				for(int f=task_W_startIdx[d]; f<task_W_endIdx[d]; f++) {
					out.write(combined_W[f]+"");
					out.newLine();
				}
				out.close();
			}
		}
		catch(Exception e) {
			System.err.println("ERROR WRITING WEIGHTS!!");
			e.printStackTrace();
		}

	}

	private void printRegularizerVal() {
		double regPart=0;
		double[] arr1 = pathwayFunc.get(0); // size: N_pathways
		double[] arr2 = pathwayFunc.get(1); // size: N_pathways
		for(int k=0; k<numPathways; k++) {
			regPart += Math.pow(arr1[k] - arr2[k], 2); // f_i - g_i
		}
		regularizer_curr = (regPart*lambdas[0]);
		System.out.println("Regularizer: "+regPart);
	}

	private double dotProd(double[] v1, double[] v2) {
		double prod=0;
		try{
			if(v1.length != v2.length)
				throw new Exception("Dot product not possible! Vectors of different lengths input!!");
			for(int l=0; l<v1.length; l++)
				prod += v1[l]*v2[l];
		}
		catch(Exception e) {
			System.out.println(e.getMessage());
			e.printStackTrace();
		}
		//System.out.println("\t\t>>>> w.dG = "+prod);
		return prod;
	}

	public void getGradG() {
		Arrays.fill(gradG, 0.0); // initialize to zero

		int lidx=0;
		for(int d=0; d<numTasks; d++) {
			double[] predVals = task_predValsStore.get(d);
			ArrayList<ArrayList<Integer>> trainFeats = task_FeatsStore.get(d); 
			ArrayList<ArrayList<Double>> trainVals = task_FeatValsStore.get(d);
			int[][] pArray = task_pathwayVectors.get(d); // dimension: nPos x N
			for(int k=0; k<numPathways; k++) {
				double term1=0, term2; 
				// does (f_k + g_k) below
				for(int dx=0; dx<numTasks; dx++) {
					term1 += (pathwayFunc.get(dx))[k];
				}
				for(int n=0; n<task_numPos.get(d); n++) {
					if(pArray[n][k] == 0)
						continue;
					ArrayList<Integer> feats = trainFeats.get(n);
					ArrayList<Double> vals = trainVals.get(n);
					for(int f=0; f<feats.size()-1; f++) { 
						double xif = vals.get(f);
						double expGrad = xif * Math.exp(predVals[n]/C)/C;
						term2 = (pArray[n][k] * expGrad); // "-1" since feature ids start at 1, arrays at 0
						gradG[task_W_startIdx[d]+feats.get(f)-1] += 2 / ((double) task_numPos.get(d)) * lambdas[lidx] * term1 * term2;
					}
				}
			}
		}
	}

	public double getvalG() {
		double res=0, regPart=0;
		int lidx=0;
		// computes \sum_i ( f_i^2 + g_i^2 + ... )
		for(int k=0; k<numPathways; k++) {
			double taskSum=0;
			for(int d=0; d<numTasks; d++) {
				double[] arr = pathwayFunc.get(d); // size: N_pathways
				taskSum += arr[k]; // f_i
			}
			regPart += taskSum*taskSum;
		}
		res = lambdas[lidx] * regPart;
		return res;
	}

	public void readData(String dataFile, int taskNum) {
		try{
			BufferedReader br = new BufferedReader(new FileReader(dataFile));
			String strLine; String[] temp;
			// find size of training file..
			int size=0;
			while ((strLine = br.readLine()) != null)   {
				size++;
			}
			br.close();
			task_trainSizes.add(size);
			System.out.println("Task-"+taskNum+" :  #Examples:"+size);

			// initialize space for training data
			ArrayList<ArrayList<Integer>> trainFeats = new ArrayList<ArrayList<Integer>>(size); // stores ids of features found in each example
			ArrayList<ArrayList<Double>> trainVals = new ArrayList<ArrayList<Double>>(size); // stores vals of features in each ex.
			br = new BufferedReader(new FileReader(dataFile));
			int idx=0, maxFeatNum=0, numPosExamples=0;
			// read in training data in libsvm format.. <label> <featid>:<featval> <featid>:<featval> ...
			while ((strLine = br.readLine()) != null)   {
				ArrayList<Integer> feats = new ArrayList<Integer>();
				ArrayList<Double> vals = new ArrayList<Double>();
				StringTokenizer st = new StringTokenizer(strLine);
				double label = Double.parseDouble(((String)st.nextElement()));
				while(st.hasMoreElements()){
					temp = ((String)st.nextElement()).split(":");
					int featNum = Integer.parseInt(temp[0]); 	
					feats.add(featNum); 
					vals.add(Double.parseDouble(temp[1]));
					if(featNum > maxFeatNum) {
						maxFeatNum = featNum;
					}
				}
				feats.add(-100); // corresponds to the label of the example
				vals.add(label);
				trainFeats.add(idx, feats);
				trainVals.add(idx, vals);
				idx++;
				if(label == POS_LABEL) {
					numPosExamples++;
				}
			}
			br.close();
			task_numPos.add(numPosExamples);
			task_numFeats.add(maxFeatNum);

			// store the data into the per-task maps
			task_FeatsStore.put(taskNum,trainFeats);
			task_FeatValsStore.put(taskNum,trainVals);

			// DEBUG
			System.out.println("[CHECK] trainFeats size: "+trainFeats.size()+"\ttrainVals size: "+trainVals.size()
					+" featsStore size:"+task_FeatsStore.size()+" valsStore size:"+task_FeatValsStore.size());
		}
		catch(Exception e) {
			System.out.println("Format Exception while reading file! "+dataFile);
			e.printStackTrace();
		}
	}

	public void readPathways(String pFile, int taskNum) {

		// initialize array
		int[][] pArray = new int[task_numPos.get(taskNum)][numPathways];
		System.err.println("[CHECK] numPos:"+task_numPos.get(taskNum)+" numPathways:"+numPathways);
		int idx=0, p=0, entry=0;
		double l2norm=0;
		try{
			BufferedReader br = new BufferedReader(new FileReader(pFile));
			String strLine;
			// read in pathways file
			while ((strLine = br.readLine()) != null)   {
				p=0;
				StringTokenizer st = new StringTokenizer(strLine, ",");
				while(st.hasMoreElements()){
					entry = Integer.parseInt((String)st.nextElement());
					pArray[idx][p++] = entry;
					l2norm += entry*entry;
				}
				idx++;
			}
			br.close();

			// store the data into the pathway map
			task_pathwayVectors.put(taskNum, pArray);
		}
		catch(Exception e) {
			System.err.println("While reading file, error! "+pFile+" at line:"+idx+" column:"+p+" entry:"+entry);
			e.printStackTrace();
		}
	}

	public double dotProdwithW(ArrayList<Integer> feats, ArrayList<Double> vals, int startIdx) {
		double dotprod=0;
		for(int f=0; f<feats.size()-1; f++) { // last feature id is "-100", corresp to label which is stored in vals.
			dotprod += combined_W[startIdx + feats.get(f)-1] * vals.get(f);	// does w.x ... feature names start from 1, weight vector starts from 0
		}
		return dotprod;
	}

	// computes classification error
	public void computeError(int startTask, int endTask) {
		for(int task=startTask; task<endTask; task++) {
			int trainTask = task%numTasks;
			ArrayList<ArrayList<Integer>> feats = task_FeatsStore.get(task);
			ArrayList<ArrayList<Double>> vals = task_FeatValsStore.get(task);

			// compute predicted values on all data instances for this task
			System.out.println("Printing test labels...");
			ArrayList<Double> pointVals;
			double numCorrect=0, numPos=0, corrPos=0, markedPos=0;
			int tot = task_trainSizes.get(task);
			for(int n=0; n<tot; n++) {
				pointVals = vals.get(n);
				double pred = dotProdwithW(feats.get(n), pointVals, task_W_startIdx[trainTask]);
				double yi = pointVals.get(pointVals.size()-1);
				pred = (pred >= 0) ? POS_LABEL : NEG_LABEL;
				// output labels on test data
				if(task >= 2*numTasks) {
					System.err.println("task"+trainTask+" "+yi+" "+pred);
				}
				if(pred == yi)
					numCorrect++;
				if(pred == POS_LABEL)
					markedPos++;
				if(yi == POS_LABEL) {
					numPos++;
					if(pred == yi)
						corrPos++;
				}

			}
			double acc = numCorrect/tot;
			double prec = 0;
			if(markedPos > 0)
				prec = corrPos/markedPos;
			double rec = corrPos/numPos;
			double f1 = 0;
			if((prec+rec) > 0)
				f1 = 2*prec*rec/(prec+rec);

			if(task < numTasks) {
				System.out.println("[TRAIN] Computing error on TRAIN data..");
			}
			else if(task >= numTasks && task < 2*numTasks) {
				System.out.println("[HO] Computing error on HELD-OUT data..");
			}
			else if(task >= 2*numTasks) {
				System.out.println("[TEST] Computing error on TEST data..");
			}
			System.out.println("[OVERALL] Task:"+trainTask+" Correct: "+numCorrect+" out of: "+tot+" Accuracy: "+acc);
			System.out.println("[POS CLASS] Task:"+trainTask+" "+corrPos+"/"+numPos+"\tP: "+prec+"\tR: "+rec+"\tF1: "+f1);
		}
	}

	public static void main (String args[]) {

		if(args.length < 1) {
			System.out.println("Usage: java MultitaskPathOpt <data-splits-file> <regParam-lambda> <regParam-sigma> <class-cost-ratio> <file-out-weight>");
			System.out.println();
			System.out.println("The first parameter is mandatory. The rest will be assigned default values if missing.");
			System.out.println("<data-splits-file> : a file containing the location of the training, test and held-out folds of all tasks.");
			System.out.println("                     Please check the provided 'sample_datasplits.txt' for an example.");
			System.out.println("<regParam-lambda>  : parameter indicating importance of the pathway regularizer term (default: 0.01)");
			System.out.println("<regParam-sigma>   : parameter that controls importance of the L2 regularizer term (default: 1)");
			System.out.println("<class-cost-ratio> : fraction indicating the pos:neg class skew in data. See the end for an example.");
			System.out.println("                     The loss terms will be computed accordingly (default: 1)");
			System.out.println("<file-out-weight>  : location for an output file to save the learned weights (default: none)");
			System.out.println();
			System.out.println("Example for assigning <class-cost-ratio>: If the data has 100 times more negatives for a single positive, set this value to 100.");
			System.exit(0);
		}

		double[] params = new double[2];
		String[] trainData_files, heldOutData_files, testData_files, pathwayFiles;
		int numTasks = 2;
		trainData_files = new String[numTasks];
		pathwayFiles = new String[numTasks];
		heldOutData_files = new String[numTasks];
		testData_files = new String[numTasks];
		String wtFilePrefix="";
		
		// set default values for parameters (used if none given)
		double classCostRatio=1;
		params[0] = 0.01; params[1] = 0.5;

		// read in data parameters from input file
		try{
			BufferedReader br = new BufferedReader(new FileReader(args[0]));
			String strLine; String[] temp;
			int c=0;
			while ((strLine = br.readLine()) != null)   {
				temp = strLine.split("\\s+");
				trainData_files[c] = temp[0];
				pathwayFiles[c] = temp[1];
				heldOutData_files[c] = temp[2];
				testData_files[c] = temp[3];
				c++;
			}
			br.close();
		}
		catch(Exception e) {
			e.printStackTrace();
		}

		// set regularization parameter
		if(args.length > 1) {
			params[0] = Double.parseDouble(args[1]);
		}
		if(args.length > 2) {
			params[1] = Double.parseDouble(args[2]);
		}
		if(args.length > 3) {
			classCostRatio = Double.parseDouble(args[3]);
		}
		if(args.length > 4) {
			wtFilePrefix = args[4];
		}

		int iters=0;
		MultitaskPathOpt dcOpt = new MultitaskPathOpt(numTasks, params, trainData_files, pathwayFiles, heldOutData_files, testData_files, classCostRatio);
		double improve=10000;
		System.out.println("[OPT] Starting optimization...");
		while(improve > 5 && iters < 10) {
			dcOpt.optimizeCvxApprox(OPT_MAX_ITER);
			dcOpt.computeTrueValue();
			improve = dcOpt.prevValTrue - dcOpt.currValTrue;
			iters++;
			System.out.println("Global Iter#"+iters+" Improvement:"+improve);
		}

		System.out.println("Finished Optimization!\nNow testing!!");
	    //dcOpt.printWeights(wtFilePrefix); // save learned weights to a file
		dcOpt.computeError(0, dcOpt.numTasks); // training data error
		dcOpt.computeError(dcOpt.numTasks, 3*dcOpt.numTasks); // held-out and test error
	}

	private class LBFGSOptimizer implements Optimizable.ByGradientValue {
		boolean gradStale, valStale;
		int iterations;
		double funcVal;
		public LBFGSOptimizer() {
			gradStale = true; valStale = true;
			iterations = 0;
			funcVal = 0;
		}

		public void getGradF() {
			int lidx=0;

			loglossGrad();// initializes gradient to all zeros, before computing loglossgrad

			l2gradient();

			for(int d=0; d<numTasks; d++) {
				double[] predVals = task_predValsStore.get(d);
				ArrayList<ArrayList<Integer>> trainFeats = task_FeatsStore.get(d); 
				ArrayList<ArrayList<Double>> trainVals = task_FeatValsStore.get(d);
				int[][] pArray = task_pathwayVectors.get(d); // dimension: nPos x N
				double update;
				for(int k=0; k<numPathways; k++) { // does outer summation of term-2 in Eqn(5)
					for(int n=0; n<task_numPos.get(d); n++) {
						if(pArray[n][k] == 0)
							continue;
						ArrayList<Integer> feats = trainFeats.get(n);
						ArrayList<Double> vals = trainVals.get(n);
						for(int f=0; f<feats.size()-1; f++) { 
							double xif = vals.get(f);
							double expGrad = xif * Math.exp(predVals[n]/C)/C;
							update = pathwayFunc.get(d)[k] * (pArray[n][k] * expGrad); // "-1" since feature ids start at 1, arrays at 0
							gradL[feats.get(f)-1+task_W_startIdx[d]] += 4 / ((double) task_numPos.get(d)) * lambdas[lidx] * update;
						}
					}
				}
			}
		}

		public double getvalF() {
			double res=0, regPart=0;
			double logl = getLogLoss();
			double l2Reg = getL2regularizer();
			//logLoss_curr = logl;
			int lidx=0;
			// computes \sum_i ( f_i^2 + g_i^2 + ... )
			for(int d=0; d<numTasks; d++) {
				double[] arr = pathwayFunc.get(d); // size: N_pathways
				for(int k=0; k<numPathways; k++) {
					regPart += arr[k]*arr[k]; // f_i^2
				}
			}
			res = logl + l2Reg + 2 * lambdas[lidx] * regPart;
			return res;
		}

		// computes approximation of loss function - Eqn (4)
		public double getValue() {
			double fnVal;
			if(!valStale) {
				return funcVal;
			}
			computePathwayFunc(); // precomputes f_i and g_i
			F_curr = getvalF();
			fnVal = F_curr + dotProd(combined_W, gradG); // gradG : constant

			System.out.println("\t\tIter#"+iterations+"\tFunction value: "+fnVal);
			funcVal = -fnVal;
			valStale = false;
			iterations++;
			return funcVal;
		}


		public void getValueGradient(double[] gradient) {
			//long startTime = System.currentTimeMillis(); //System.out.println("[OPT] Getting gradient values..");
			if(gradStale) {
				getGradF();
				addGradG();
			}
			for(int p=0; p<gradL.length; p++)
				gradient[p] = -gradL[p];
			//System.out.println("\t\t[Gradient] Time taken: "+(System.currentTimeMillis()-startTime));
			gradStale = false;
		}

		// The following get/set methods satisfy the Optimizable interface
		public int getNumParameters() { return numParams; }
		public double getParameter(int i) { return combined_W[i]; }
		public void getParameters(double[] buffer) {
			System.arraycopy(combined_W, 0, buffer, 0, combined_W.length);
		}

		public void setParameter(int i, double r) {
			//System.out.println("[OPT] Setting single parameter");
			combined_W[i] = r;
			setGradStale();
		}
		public void setParameters(double[] newParameters) {
			//System.out.println("[OPT] Setting parameters");
			System.arraycopy(newParameters, 0, combined_W, 0, newParameters.length);
			setGradStale();
		}

		public void setGradStale() {
			gradStale = true;
			valStale = true;
		}

		public void addGradG() {
			for(int idx=0; idx<gradL.length; idx++)
				gradL[idx] += gradG[idx];
		}

		// returns logloss over all tasks
		public double getLogLoss() {
			ArrayList<ArrayList<Double>> dataVals;
			double[] predVals;
			ArrayList<Double> vals;	// temporary variable

			double totLogloss=0, cost=1;
			for(int i=0; i<numTasks; i++) {
				predVals = task_predValsStore.get(i); // stores w.x for all data, for all domains
				dataVals = task_FeatValsStore.get(i);
				double logloss=0;
				for(int n=0; n<task_trainSizes.get(i); n++) {
					vals = dataVals.get(n);
					double yi = vals.get(vals.size()-1);	// label of the data
					double wxi = predVals[n];	// w.x
					cost=1;
					if(yi==POS_LABEL)
						cost = costRatio;
					logloss += cost * Math.log(1 + Math.exp(-wxi*yi));
				}
				double currSize = 1; //trainSizes.get(i);
				totLogloss += logloss/currSize;
				//System.out.println("\t\tLogloss Task:"+i+"= "+logloss);
			}
			//System.out.println("\n\t\tTotal log loss: "+totLogloss);
			return totLogloss;
		}

		// computes gradient of log-loss part.. initializes gradL
		// feature numbers start from 1, gradient array starts from 0
		public void loglossGrad() {
			Arrays.fill(gradL, 0.0);
			double cost=1;
			for(int d=0; d<numTasks; d++) {
				ArrayList<ArrayList<Integer>> trainFeats = task_FeatsStore.get(d);
				ArrayList<ArrayList<Double>> trainVals = task_FeatValsStore.get(d);
				double[] predVals = task_predValsStore.get(d);
				//int sizeOfTask=task_trainSizes.get(d);

				for(int n=0; n<task_trainSizes.get(d); n++) {
					ArrayList<Integer> feats = trainFeats.get(n);
					ArrayList<Double> vals = trainVals.get(n);
					double yi = vals.get(vals.size()-1);
					cost=1;
					if(yi==1)
						cost = costRatio;
					double z = -(predVals[n] * yi);
					double sigmoid = 1.0/(1.0 + Math.exp(-z));
					for(int f=0; f<feats.size()-1; f++) { // last entry is the label
						double update = cost * -vals.get(f) * sigmoid * yi; // - n1/(n1+n2) * xy * sigmoid(-wxy)
						gradL[task_W_startIdx[d] + feats.get(f)-1] += update; // divide by sizeOfTask if you want to;
					}
				}
			}
		}

		// l2 to prevent overfitting
		public double getL2regularizer() {
			double sumAll=0;
			double sigma=lambdas[1];
			//System.out.println("Sigma: "+lambdas[lidx]);
			for(int d=0; d<numTasks; d++) {
				double sum=0;
				for(int f=task_W_startIdx[d]; f<task_W_endIdx[d]; f++) {
					sum += combined_W[f]*combined_W[f];
				}
				sumAll += sigma * sum;
			}
			return sumAll;
		}

		// gradient of l2
		public void l2gradient() {
			double sigma=lambdas[1];
			for(int d=0; d<numTasks; d++) {
				for(int f=task_W_startIdx[d]; f<task_W_endIdx[d]; f++) {
					gradL[f] += sigma * 2 * combined_W[f];
				}
			}
		}
	};
}

