package distance;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.OutputStreamWriter;
import java.io.Reader;
import java.util.List;

import be.ac.ulg.montefiore.run.jahmm.Hmm;
import be.ac.ulg.montefiore.run.jahmm.ObservationVector;
import be.ac.ulg.montefiore.run.jahmm.OpdfMultiGaussian;
import be.ac.ulg.montefiore.run.jahmm.OpdfMultiGaussianFactory;
import be.ac.ulg.montefiore.run.jahmm.io.ObservationSequencesReader;
import be.ac.ulg.montefiore.run.jahmm.io.ObservationVectorReader;
import be.ac.ulg.montefiore.run.jahmm.toolbox.KullbackLeiblerDistanceCalculator;

import weka.core.Instance;

/**
 * Implementation of Rabiner's symmetrized Hidden Markov Model
 * distance measure between two sequences.
 * 
 * @author Shalisa Pattarawuttiwong
 */
public class HMMDistance {
	
	/**
	 * Convert an instance to an ObservationVector list.
	 * 
	 * @param inst Instance
	 * @return An List<ObservationVector>
	 * @throws Exception
	 */
	public List<ObservationVector> instanceToObservation(Instance inst) throws Exception{
		// convert instance to string: [1, 2, 3] -> "[1 2 3]"
		// String instString = Arrays.toString(inst.toDoubleArray()).replace(",","");
		double[] instArray = inst.toDoubleArray();
		
		// write out to a .seq file
		File f_out = new File("./data/instance1.seq");
		FileOutputStream fos = new FileOutputStream(f_out);
		BufferedWriter bw = new BufferedWriter(new OutputStreamWriter(fos));
		
		for (int i = 0; i < instArray.length; i = i + 2) {
			bw.write("[");
			bw.write(Double.toString(instArray[i]));
			bw.write(" ");
			bw.write(Double.toString(instArray[i+1]));
			bw.write("] ; ");
		}
		bw.write("\n");
		bw.close();
		
		// read in file to convert to observations
		Reader reader = new FileReader("./data/instance1.seq");
		List<List<ObservationVector>> obs =
				ObservationSequencesReader.readSequences(
						new ObservationVectorReader(), reader);
		reader.close();
		return obs.get(0);
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
		
		for (int i = 0; i < obs.size(); i++) {
			ObservationVector tup = obs.get(i);
			for (int idx = 0; idx < 2; idx++) {
				mean[idx] += tup.value(idx);
			}
		}
		for (int i = 0; i < 2; i++) {
			mean[i] = mean[i]/obs.size();
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
		for (int i = 0; i < obs.size(); i++) {
			cov += (obs.get(i).value(x) - mean[x]) * (obs.get(i).value(y) - mean[y]);
		}
		return cov / (obs.size());		
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
		double[][] covMat = new double[2][2];
		for (int i = 0; i < 2; i++) {
			for (int j = 0; j < 2; j++) {
				double cov = calcCovariance(obs, mean, i, j);
				covMat[i][j] = cov;
				covMat[j][i] = cov;
			}
		}
		return covMat;
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
	public Hmm<ObservationVector> initMultiHMM(Instance x, int states) throws Exception{
		List<ObservationVector> obs = instanceToObservation(x);
		// generates a new gaussian distribution with 
		// mean and covariance matrices
		double[] mean = calcVectorMean(obs);
		double[][] covariance = calcCovarianceMatrix(obs, mean);
		
		Hmm<ObservationVector> hmm = new Hmm<ObservationVector>(
				states, new OpdfMultiGaussianFactory(states));
		
		OpdfMultiGaussian omg = new OpdfMultiGaussian(mean, covariance);
		// generates 10,000 observation vectors according to
		// distribution
		ObservationVector[] obsNew = new ObservationVector[10000];
		for (int i = 0; i < obsNew.length; i++)
			obsNew[i] = omg.generate();
		
		// find gaussian distribution that fits observations
		omg.fit(obsNew);
		
		for (int s = 0; s < states; s++) {
			// prob state is initial
			hmm.setPi(s, 1.0/states * states);
			hmm.setOpdf(s, omg);
			// uniform matrix
			for (int s_prime = 0; s_prime < states; s_prime++) {
				hmm.setAij(s, s_prime, 1.0/states);
			}
		}
		
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
	public double distance(Instance x, 
			Instance y, int states) throws Exception {
		// initializes HMMs for x and y
		Hmm<ObservationVector> xHmm = initMultiHMM(x, states);
		Hmm<ObservationVector> yHmm = initMultiHMM(y, states);
		
		KullbackLeiblerDistanceCalculator kld = 
				new KullbackLeiblerDistanceCalculator();
		return (kld.distance(xHmm, yHmm) + kld.distance(yHmm, xHmm) / 2.0);
	}
	
	/**
	 * Constructor for HMMDistance.
	 */
	public HMMDistance() {
	}

}
