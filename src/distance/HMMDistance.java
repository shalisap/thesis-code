package distance;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import clustering.KMeans;
import be.ac.ulg.montefiore.run.jahmm.Hmm;
import be.ac.ulg.montefiore.run.jahmm.ObservationVector;
import be.ac.ulg.montefiore.run.jahmm.OpdfMultiGaussian;
import be.ac.ulg.montefiore.run.jahmm.OpdfMultiGaussianFactory;
import be.ac.ulg.montefiore.run.jahmm.toolbox.KullbackLeiblerDistanceCalculator;
import weka.core.Attribute;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;

/**
 * Implementation of Rabiner's symmetrized Hidden Markov Model
 * distance measure between two sequences.
 * 
 * @author Shalisa Pattarawuttiwong
 */
public class HMMDistance extends AbstractDistance {
	
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
	 * Minimum standard deviation for a state in the clustering
	 * phase. Anything less than this leaves log likelihood
	 * prone to underflow errors.
	 */
	private static final double EPSILON = 0.47; 
    
	/**
	 * Constructor for HMMDistance.
	 */
	public HMMDistance(Instance a, Instance b, int m) {
         this.states = m;
         this.x = a;
         this.y = b;
    }
	
	/**
	 * Constructor for HMMDistance.
	 */
	public HMMDistance() {
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
	 * Convert an instance to an ObservationVector list.
	 * 
	 * @param inst Instance
	 * @return An List<ObservationVector>
	 * @throws Exception
	 */
	public List<ObservationVector> instanceToObservation(Instance inst) {
		List<ObservationVector> obs = new ArrayList<ObservationVector>();
		// get values of instance
		double[] instArray = inst.toDoubleArray();
		
		double[] o_arr = new double[2]; 
		for (int i = 0; i < instArray.length; i = i + 2) {
			o_arr[0] = instArray[i];
			o_arr[1] = instArray[i+1];
			// convert each pair to observation
			obs.add(new ObservationVector(o_arr));
		}
		return obs;
	}
	
	/**
	 * Assuming only two dimensions, calculate the vector mean
	 * given an ObservationVector list, obs, where
	 * obs = [(in_0, out_0), (in_1, out_1), ..., (in_{n-1}, out_{n-1})]:
	 * 
	 * X = [mean(in_0, in_1,...,in_{n-1})), mean(out_0, out_1, ..., out_{n-1})]
	 * 
	 * @param obs 
	 * @return the vector mean of obs
	 */
	public double[] calcVectorMean(List<ObservationVector> obs) {
		double[] mean = new double[2];
		double num = obs.size()*2;
		
		for (int i = 0; i < obs.size(); i++) {
			ObservationVector tup = obs.get(i);
			for (int idx = 0; idx < 2; idx++) {
				double val = tup.value(idx);
				if (val == -1.0) {
					num -= 1.0;
				} else {
					mean[idx] += tup.value(idx);
				}
			}
		}
		for (int i = 0; i < 2; i++) {
			mean[i] = mean[i]/(num/2);
		}
		
		return mean;
	}
	
	/**
	 * Calculates the covariance given the ObservationVector list, 
	 * obs = [(in_0, out_0), (in_1, out_1), ..., (in_{n-1}, out_{n-1})], 
	 * mean vector of obs, and the indices being calculated:
	 * 
	 * COV = (sum from i = 0 to n - 1 (in_i - mean(in))(out_i - mean(out))) / n 
	 * 
	 * @param obs List<ObservationVector>
	 * @param mean Mean vector of obs
	 * @param x Index x 
	 * @param y Index y 
	 * @return covariance between two given indices
	 */
	public double calcCovariance(List<ObservationVector> obs, double[] mean, int x, int y) {
		double cov = 0;
		double num = obs.size()*2;
		for (int i = 0; i < obs.size(); i++) {
			cov += (obs.get(i).value(x) - mean[x]) * (obs.get(i).value(y) - mean[y]);
		}
		return cov / (num/2);		
	}
	
	/**
	 * Calculates the covariance matrix given the ObservationVector list,
	 *  obs, with length n, and the mean vector of obs.
	 * 
	 * @param obs List<ObservationVector>
	 * @param mean Mean vector of obs
	 * @return covariance matrix of obs
	 */
	public double[][] calcCovarianceMatrix(List<ObservationVector> obs, double[] mean) {
		
		// remove -1's
		List<ObservationVector> filtObs = new ArrayList<ObservationVector>();
		for (int i = 0; i < obs.size(); i++) {
			if (obs.get(i).values()[0] != -1.0 &&
					obs.get(i).values()[1] != -1.0) {
				filtObs.add(obs.get(i));
			}
		}
		
		double[][] covMat = new double[2][2];
		for (int i = 0; i < 2; i++) {
			for (int j = 0; j < 2; j++) {
				double cov = Math.max(calcCovariance(filtObs, mean, i, j), EPSILON);
				covMat[i][j] = cov;
				covMat[j][i] = cov;
			}
		}
		return covMat;
	}
	
	/**
	 * Given a set of cluster labels, partitions x into k clusters.
	 * @param x Instances
	 * @param labels cluster labels for x
	 * @param k number of clusters
	 * @return An array of Instances, each of which contains the
	 * 		series from one cluster, where the series within 
	 * 		clusters are concatenated.
	 */
	private ArrayList<Instance> partition(Instances x, int[] labels, int k) {
		ArrayList<Instance> part = new ArrayList<Instance>();
		
		double[][] vals = new double[k][];
		Map<Integer,Integer> occur = new HashMap<Integer,Integer>();
		// figure out occurrences of each label
		for (int l: labels){
			if (occur.containsKey(l)) {
				occur.put(l, occur.get(l) + 2);
			} else {
				occur.put(l, 2);
			}
		}
		
		// initialize arrays
		for (int i = 0; i < k; i++) {
			part.add(new Instance(occur.get(i)));
			vals[i] = new double[occur.get(i)];
		}
		
		// partition according to labels
		for (int i = labels.length - 1; i >= 0; i--) {
			for (int j = x.instance(i).numAttributes() - 1; j >= 0; j--) {
				vals[labels[i]][occur.get(labels[i]) - 1] = x.instance(i).value(j);
				occur.put(labels[i], occur.get(labels[i]) - 1);
			}
		}
		
		// add to instances
		for (int i = 0; i < part.size(); i++) {
			part.get(i).replaceMissingValues(vals[i]);
		}
		return part;
	}
	
	/**
	 * Generates a multivariate gaussian distribution function
	 * from an instance
	 * 
	 * @param cluster Instance
	 * @return A multivariate gaussian distribution
	 */
    private OpdfMultiGaussian calcGaussian(Instance cluster) {
		
		List<ObservationVector> obs = instanceToObservation(cluster);
		// generates a new gaussian distribution with 
		// mean and covariance matrices
		double[] mean = calcVectorMean(obs);
		double[][] covariance = calcCovarianceMatrix(obs, mean);

		OpdfMultiGaussian omg = new OpdfMultiGaussian(mean, covariance);
		// generates 10,000 observation vectors according to
		// distribution
		ObservationVector[] obsNew = new ObservationVector[10000];
		for (int i = 0; i < obsNew.length; i++)
			obsNew[i] = omg.generate();
		
		// find gaussian distribution that fits observations
		omg.fit(obsNew);
		return omg;
    }
	
	/**
	 * Given an Instance, initializes an HMM with 
	 * uniform Pi and A parameters and calculates
	 * a gaussian distribution that fits x for B.
	 * 
	 * @param x Instance
	 * @param states number of states of the generated HMM
	 * @return HMM<ObservationVector>
	 */
	public Hmm<ObservationVector> initMultiHMM(Instance x, int states) {
		// change instance x to to instances of (IN, OUT)
		FastVector attInfo = new FastVector();
		attInfo.addElement(new Attribute("IN", 0));
		attInfo.addElement(new Attribute("OUT", 0));
		
		Instances xInsts = new Instances("cellCount",attInfo, x.numAttributes());
		for (int a = 0; a < x.numAttributes(); a = a + 2) {
			Instance i = new Instance(2);
			i.setValue(0, x.value(a));
			i.setValue(1, x.value(a + 1));
			xInsts.add(i);
		}
		//System.out.println(xInsts.toString());
		
		// cluster x into m clusters with kmeans
        ManhattanDistance manD = new ManhattanDistance();
        DistanceFunction manDist = manD;
        KMeans kmeans = new KMeans(xInsts, manDist);
        kmeans.setNumClusters(states-1);
        kmeans.setNumIterations(100);
        kmeans.cluster();        
        int[] labels = kmeans.getClusters();
        //System.out.println(Arrays.toString(labels));
        
        ArrayList<Instance> part = partition(xInsts, labels, states-1); 
        
		Hmm<ObservationVector> hmm = new Hmm<ObservationVector>(
				states, new OpdfMultiGaussianFactory(2));
        
        // for each cluster, calculate gaussian emission distributions
        List<OpdfMultiGaussian> Bs = new ArrayList<OpdfMultiGaussian>();
        for (Instance c: part) {
            OpdfMultiGaussian opdf = calcGaussian(c);
            Bs.add(opdf);
        }
        
        // gaussian for (-1, -1) destroy state
        double[] desMean = new double[]{-1.0, -1.0};
        double[][] desCov = new double[][]{{EPSILON, EPSILON}, {EPSILON, EPSILON}};
		OpdfMultiGaussian desOpdf = new OpdfMultiGaussian(desMean, desCov);
		// generates 10,000 observation vectors according to
		// distribution
		ObservationVector[] obsNew = new ObservationVector[10000];
		for (int i = 0; i < obsNew.length; i++)
			obsNew[i] = desOpdf.generate();
		desOpdf.fit(obsNew);
        
		for (int s = 0; s < states-1; s++) {
			// prob state is initial
			hmm.setPi(s, (1.0/(states-1)) * (states-1));
			hmm.setOpdf(s, Bs.get(s));
			// uniform matrix
			for (int s_prime = 0; s_prime < states; s_prime++) {
				hmm.setAij(s, s_prime, 1.0/states);
			}
		}
		// destroy state
		hmm.setPi(states-1, 0);
		hmm.setOpdf(states-1, desOpdf);
		for (int s_prime = 0; s_prime < states-1; s_prime++) {
			hmm.setAij(states-1, s_prime, 0);
		}
		hmm.setAij(states-1, states-1, 1);
		
		return hmm;
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
		Hmm<ObservationVector> xHmm = initMultiHMM(x, states);
		Hmm<ObservationVector> yHmm = initMultiHMM(y, states);
		
		KullbackLeiblerDistanceCalculator kld = 
				new KullbackLeiblerDistanceCalculator();
		return (kld.distance(xHmm, yHmm) + kld.distance(yHmm, xHmm) / 2.0);
	}

}
