package distance;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import clustering.KMeans;
import be.ac.ulg.montefiore.run.jahmm.Hmm;
import be.ac.ulg.montefiore.run.jahmm.Observation;
import be.ac.ulg.montefiore.run.jahmm.ObservationInteger;
import be.ac.ulg.montefiore.run.jahmm.OpdfInteger;
import be.ac.ulg.montefiore.run.jahmm.OpdfIntegerFactory;
import be.ac.ulg.montefiore.run.jahmm.learn.BaumWelchLearner;
import be.ac.ulg.montefiore.run.jahmm.toolbox.MarkovGenerator;
import weka.core.Attribute;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;

public class DiscreteHMMDistance extends AbstractDistance {
	
    /**
     * The number of HMM states to generate.
     */
    protected int states;
    
    /**
     * Instance 
     */
    protected Instance x;
    
    /**
     * Instance
     */
    protected Instance y;
    
	/**
	 * Constructor for HMMDistance.
	 */
	public DiscreteHMMDistance(Instance a, Instance b, int m) {
         this.states = m;
         this.x = a;
         this.y = b;
    }
	
	/**
	 * Constructor for HMMDistance.
	 */
	public DiscreteHMMDistance() {
    }
    
    /**
     * Set the number of HMM states to generate
     * @param m Number of states
     */
    public void setNumStates(int m)
    		throws IllegalArgumentException{
    	if (m <= 0) {
    		throw new IllegalArgumentException("Cannot set the number "
    				+ "of states to fewer than 1");
    	} else this.states = m;
    }
    
    /**
     * Get the number of HMM states to generate
     * @return m Number of states
     */
    public int getNumStates() {
    	return this.states;
    }

	/**
	 * Given a set of cluster labels, partitions x into k clusters.
	 * @param x Instances
	 * @param labels cluster labels for x
	 * @param k number of clusters
	 * @param multiToDiscrete mapping of symbol to integer key
	 * @return An array of Instances, each of which contains the
	 * 		series from one cluster, where the series within 
	 * 		clusters are concatenated.
	 */
	private int[][] partition(Instances x, int[] labels, int k, 
			HashMap<String, Integer> multiToDiscrete) {
		
		ArrayList<Instance> part = new ArrayList<Instance>();
		
		int[][] vals = new int[k][];
		Map<Integer,Integer> occur = new HashMap<Integer,Integer>();
		// figure out occurrences of each label
		for (int l: labels){
			if (occur.containsKey(l)) {
				occur.put(l, occur.get(l) + 1);
			} else {
				occur.put(l, 1);
			}
		}
		
		// initialize arrays
		for (int i = 0; i < k; i++) {
			part.add(new Instance(occur.get(i)));
			vals[i] = new int[occur.get(i)];
		}
		
		// partition according to labels
		for (int i = labels.length - 1; i >= 0; i--) {
			double in = x.instance(i).value(0);
			double out = x.instance(i).value(1);
			String key = Arrays.toString(new double[]{in, out});
			vals[labels[i]][occur.get(labels[i]) - 1] = 
					multiToDiscrete.get(key);
			occur.put(labels[i], occur.get(labels[i]) - 1);
		}
		return vals;
	}
    
	/**
	 * Maps symbols (pairs of (in,out) cell counts) to integers.
	 * @param x Instance 
	 * @param y Instance 
	 * @return HashMap for strings to integers
	 */
	private HashMap<String,Integer> symbolsToIntegers(Instance x, Instance y) {
		// map unique symbols to integers
        HashMap<String, Integer> multiToDiscrete = 
        		new HashMap<String, Integer>();
        
        // unique symbols and sort
		double[] arrayX = x.toDoubleArray();
		double[] arrayY = y.toDoubleArray();
		double[] arrayBoth = new double[arrayX.length + arrayY.length];
		System.arraycopy(arrayX, 0, arrayBoth, 0, arrayX.length);
		System.arraycopy(arrayY, 0, arrayBoth, arrayX.length, arrayY.length);
	    
		double[][] pairsArray = new double[arrayBoth.length/2][2];
	    for (int i = 0; i < arrayBoth.length; i += 2) {
	    	pairsArray[i/2][0] = arrayBoth[i];
	    	pairsArray[i/2][1] = arrayBoth[i+1];
	    }

	    // sort symbols from increasing to decreasing
		Arrays.sort(pairsArray, new java.util.Comparator<double[]>() {
		    public int compare(double[] a, double[] b) {
		        return Double.compare(a[0], b[0]);
		    }
		});
		
		// map each unique symbol to integer
        int disVal = 0;
        for (int i = 0; i < pairsArray.length; i++) {
        	String pair = Arrays.toString(pairsArray[i]);
        	if (!(multiToDiscrete.containsKey(pair))) {
        		multiToDiscrete.put(pair,disVal);
            	disVal += 1;
        	} 
        }	
        return multiToDiscrete;
	}
	
	/**
	 * Clusters the symbols (excluding destroy states) with k-means
	 * and manhattan distance
	 * @param x Instance
	 * @param states Number of clusters/states (k)
	 * @param multiToDiscrete HashMap from strings (symbols) to integers 
	 * @return k clusters 
	 */
	private int[][] smythInitClusters(Instance x, int states, 
			HashMap<String,Integer> multiToDiscrete) {
		// change instance x to to instances of (IN, OUT)
		FastVector attInfo = new FastVector();
		attInfo.addElement(new Attribute("IN", 0));
		attInfo.addElement(new Attribute("OUT", 0));
		
		Instances xInsts = new Instances("cellCount",attInfo, 
				x.numAttributes());
		for (int a = 0; a < x.numAttributes(); a = a + 2) {
			Instance i = new Instance(2);
			if (x.value(a) != -1 && x.value(a+1) != -1) {
				i.setValue(0, x.value(a));
				i.setValue(1, x.value(a + 1));
				xInsts.add(i);
			}
		}
		//System.out.println("Instances: " + xInsts.toString());
	
		// cluster x into m clusters with kmeans
        ManhattanDistance manD = new ManhattanDistance();
        DistanceFunction manDist = manD;
        KMeans kmeans = new KMeans(xInsts, manDist);
        kmeans.setNumClusters(states-1);
        kmeans.setNumIterations(100);
        kmeans.cluster();        
        int[] labels = kmeans.getClusters();
        //System.out.println(Arrays.toString(labels));
        
        // labels from clusters -> values in mapping
        return partition(xInsts, labels, states-1, multiToDiscrete); 
	}
	
	/* Generate several observation sequences using a HMM */
	static <O extends Observation> List<List<O>> generateSequences(Hmm<O> hmm)
	{
	  MarkovGenerator<O> mg = new MarkovGenerator<O>(hmm);
	  List<List<O>> sequences = new ArrayList<List<O>>();

	  for (int i = 0; i < 200; i++)
	    sequences.add(mg.observationSequence(100));

	  return sequences;
	}
	
	/**
	 * Given an Instance, initializes an HMM (with Smyth's
	 * initialization) uniform Pi and A parameters and calculates
	 * a distribution that fits x for B.
	 * 
	 * @param x Instance
	 * @param states number of states of the generated HMM
	 * @return HMM<ObservationInteger>
	 */
	private Hmm<ObservationInteger> initHMM(Instance x, int states
			,HashMap<String,Integer> multiToDiscrete) {
        
		// map unique symbols to integers
		//HashMap<String, Integer> multiToDiscrete = symbolsToIntegers(x, y); 
        int disVal = multiToDiscrete.size();
	
        // labels from clustering with values in mapping
        int[][] part = smythInitClusters(x, states, multiToDiscrete); 
         
        // for each cluster, calculate distribution of symbols
        double[][] bs = new double[states-1][disVal];

        for (int i = 0; i < part.length; i++) {

        	// calculates the number of times each value is seen
        	double[] b = new double[disVal];
        	int clustLen = part[i].length;
        	for (int j = 0; j < clustLen; j++) {
        		b[part[i][j]] += 1;
        	}
        
        	// calc average
        	for (int j = 0; j < disVal; j++) {
        		b[j] /= clustLen;
        	}
        	bs[i] = b;
        }
        
        
        // alphabet
        OpdfIntegerFactory factory = new OpdfIntegerFactory(disVal);
        // init new hmm
        Hmm<ObservationInteger> hmm = new 
        		Hmm<ObservationInteger>(states, factory);
        
        // for (-1, -1) destroy state
        double[] prob = new double[disVal];
        prob[0] = 1;
        for (int i = 1; i < prob.length; i++) {
            	prob[i] = 0;
        }
        
		// destroy state = hmm state 0
		hmm.setPi(0, 0);
		hmm.setOpdf(0, new OpdfInteger(prob));
		hmm.setAij(0, 0, 1);
		for (int s_prime = 1; s_prime < states; s_prime++) {
			hmm.setAij(0, s_prime, 0);
		}
		
		for (int s = 1; s < states; s++) {
			// prob state is initial is uniform
			hmm.setPi(s, (1.0/(states-1)));
			hmm.setOpdf(s, new OpdfInteger(bs[s-1]));
			// uniform matrix
			for (int s_prime = 0; s_prime < states; s_prime++) {
				hmm.setAij(s, s_prime, 1.0/states);
			}
		}
		
		BaumWelchLearner bwl = new BaumWelchLearner();
		Hmm<ObservationInteger> learntHmm = 
				bwl.learn(hmm, generateSequences(hmm));
		//System.out.println(learntHmm.toString());
		return learntHmm;
	}
	
    /**
     * Taken from the jahmm library,
     * Computes the Kullback-Leibler distance between two HMMs.
     * Edited for NaN valued errors.
     *
     * @param hmm1 The first HMM against which the distance is computed.
     *             The distance is mesured with regard to this HMM (this must
     *             be defined since the Kullback-Leibler distance is not
     *             symetric).
     * @param hmm2 The second HMM against which the distance is computed.
     * @return The distance between <code>hmm1</code> and <code>hmm2</code> with
     *      regard to <code>hmm1</code>
     */
	
    private <O extends Observation> double 
    kldistance(Hmm<O> hmm1, Hmm<? super O> hmm2) {                      
        int sequencesLength = 1000;
        int nbSequences = 10;
        double distance = 0.;
            
        for (int i = 0; i < nbSequences; i++) {
                
                List<O> oseq = new MarkovGenerator<O>(hmm1).
                observationSequence(sequencesLength);

                distance += (new ForwardBackwardNaNCalculator(oseq, hmm1).
                                lnProbability() -
                                new ForwardBackwardNaNCalculator(oseq, hmm2).
                                lnProbability()) / sequencesLength;
        }
        
        return distance / nbSequences;
    }
    
	/**
	 * Calculates the symmetrized Hidden Markov Model 
	 * distance between two Instances using jahmm's
	 * KullbackLeiblerDistanceCalculator class.
	 * 
	 * @param x Instance
	 * @param y Instance
	 * @return The distance between x and y
	 */
	//@Override 
	public double distance(Instance x, Instance y) {
		// initializes HMMs for x and y
		HashMap<String, Integer> toDiscrete = symbolsToIntegers(x, y); 
		Hmm<ObservationInteger> xHmm = initHMM(x, states, toDiscrete);
		Hmm<ObservationInteger> yHmm = initHMM(y, states, toDiscrete);
		
		double distxy = kldistance(xHmm,yHmm);
		double distyx = kldistance(yHmm,xHmm);

		return (distxy + distyx) / 2.0;
	}

}
