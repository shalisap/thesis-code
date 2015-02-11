package evaluation;

import java.util.ArrayList;

import weka.core.Instances;

/**
 * @author Shalisa Pattarawuttiwong
 *
 */
public class CollapsedPairs extends AbstractEvaluation implements Evaluation {

	protected Instances data;
	protected int[] clusterAlg;
	protected int[] groundTruth;
	protected int numEvals;
	
	/**
	 * Returns the total number of clusters for an assignment
	 * @param clusters
	 * @return number of clusters
	 */
	public int getNumClusters(int[] clusters) {
		ArrayList<Integer> numClusterList = new ArrayList<Integer>();
		int numClusters = 0;
		for (int c: clusters) {
			if (!numClusterList.contains(c)) {
				numClusterList.add(c);
				numClusters++;
			}
		}
		return numClusters;
	}
	
	/** 
	 * a = the number of pairs that are distinguished in 
	 * ground truth, and collapsed in the clustering algorithm
	 * solution.
	 * 
	 * collapsed pairs = a / total number of pairs
	 * 
	 * An asymmetric distance.
	 */
	@Override
	public double evaluate() {
		int numCollapsed = 0;
    	for (int i = 0; i < groundTruth.length; i++) {
    		for (int j = 0; j < groundTruth.length; j++) {
    			if (groundTruth[i] != groundTruth[j] && clusterAlg[i] == clusterAlg[j]) {
    				numCollapsed++;
    			}
    		}
    	}
		return numCollapsed / ((1/2) * (groundTruth.length - 1) * groundTruth.length);
	}

	/**
	 * Constructor for CollapsedPairs
	 */
	public CollapsedPairs(int n, Instances d, int[] ca, int[] gt) {
		this.data = d;
		
		// SHOULD THIS BE ALLOWED???
		if (getNumClusters(ca) != getNumClusters(gt)) {
    		throw new IllegalArgumentException("Clusters cannot have"
    				+ "different number of total clusters");
		}
		this.clusterAlg = ca;
		this.groundTruth = gt;
		
		if (n <= 0) {
    		throw new IllegalArgumentException("The number of times evaluated"
    				+ "cannot be fewer than 1.");
		}
		this.numEvals = n;
	}
}