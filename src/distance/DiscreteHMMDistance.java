package distance;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;

import be.ac.ulg.montefiore.run.jahmm.Hmm;
import be.ac.ulg.montefiore.run.jahmm.Observation;
import be.ac.ulg.montefiore.run.jahmm.ObservationInteger;
import be.ac.ulg.montefiore.run.jahmm.OpdfInteger;
import be.ac.ulg.montefiore.run.jahmm.OpdfIntegerFactory;
import be.ac.ulg.montefiore.run.jahmm.learn.BaumWelchLearner;
import be.ac.ulg.montefiore.run.jahmm.toolbox.MarkovGenerator;
import weka.core.Instance;

/**
 * Implementation of HMMDistance with 
 * a naive initialization approach.
 * 
 * @author Shalisa Pattarawuttiwong
 *
 */
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
	    	// pairs of (in,out)
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
         
        double[] xInst = x.toDoubleArray();
        // convert to discrete integers
        int[] instInteger = new int[xInst.length/2];
        double[] bs = new double[disVal];
        // make distribution uniform among symbols that exist in x.
        int idx = 0;
        int sum = 0;
        for (int i = 0; i < xInst.length; i += 2) {
        	String pair = Arrays.toString(new double[]{xInst[i],xInst[i+1]});
        	int symbol = multiToDiscrete.get(pair);
        	instInteger[idx] = symbol;
        	// don't take destroy state into distribution calculations
        	if (symbol != 0) {
            	bs[symbol] += 1;
            	sum += 1;
        	}
        	idx += 1;
        }
                
        // divide number of occurences of specific symbol
        // by total num (omitting destroy)
        for (int i = 0; i < bs.length; i++) {
        	bs[i] /= sum;
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
			hmm.setOpdf(s, new OpdfInteger(bs));
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
        int sequencesLength = 500;
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
