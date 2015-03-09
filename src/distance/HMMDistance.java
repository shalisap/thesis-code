package distance;

import be.ac.ulg.montefiore.run.jahmm.Hmm;
import be.ac.ulg.montefiore.run.jahmm.ObservationInteger;
import be.ac.ulg.montefiore.run.jahmm.toolbox.KullbackLeiblerDistanceCalculator;

/**
 * Implementation of Rabiner's symmetrized Hidden Markov Model
 * distance measure between two sequences.
 * 
 * @author Shalisa Pattarawuttiwong
 */
public class HMMDistance {
	
	/**
	 * Calculates the symmetrized distance between two 
	 * Hidden Markov Models using jahmm's
	 * KullbackLeiblerDistanceCalculator class.
	 * 
	 * @param hmm1
	 * @param hmm2 
	 * @return The distance between seq1 and seq2
	 */
	public double distance(Hmm<ObservationInteger> hmm1, 
			Hmm<ObservationInteger> hmm2) throws Exception {
		KullbackLeiblerDistanceCalculator kld = 
				new KullbackLeiblerDistanceCalculator();
		return (kld.distance(hmm1, hmm2) / 2.0);
	}
	
	/**
	 * Constructor for HMMDistance.
	 */
	public HMMDistance() {
	}

}
