package evaluation;

import java.util.*;

import weka.core.Instance;
import weka.core.Instances;
/**
 * @author Shalisa Pattarawuttiwong
 *
 */
public class DistinguishingPairs extends AbstractEvaluation implements
		Evaluation {

	protected Instances data;
	protected int[] cluster1;
	protected int[] cluster2;
	protected int numEvals;
	
	private int same;
	private int diff;
	private int same1diff2;
	private int diff1same2;
	
	private boolean adjRandIndex = false;
	
	private void setAdjRandIndex(boolean a) {
		adjRandIndex = a;
	}
	
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
	 * The Rand Index, As described in Rand (1971), given N points, X1, X2, ... , XN, and
	 * two clusterings of them Y = {Y1 ... YK1} and Y' = {Y1' ... YK1'}, 
	 * the similarity of two clusterings, c(Y, Y') = sum deltaij / n choose 2 
	 * from i < j to N. deltaij = 1 if there exist k and k' such that both 
	 * Xi and Xj are in both Yk and Y'k', 1 if there exist k and k' such that 
	 * Xi is in both Yk and Y'k' while Xj is in neither Yk or Y'k', and 0 otherwise.
	 * 
	 * Simplified: 
	 * a = number of pairs of elements in N that are in the same set in X and Y
	 * b = number of pairs of elements in N that are in the same set in X and 
	 * 		different sets in Y
	 * c = number of pairs of elements in N that are in different sets in X and in the 
	 * 		same set in Y.
	 * d = number of pairs of elements in N that are in different sets in X and 
	 * 		different sets in Y.
	 * 
	 * c(Y, Y') = a + d / a + b + c + d = a + d / (N choose 2)
	 * 
	 * c = 0 when two clusterings have no similarities -> 1 where they're identical.
	 * 
	 * the Adjusted Rand Index = (n choose 2)(a + d) - [(a+b)(a+c) + (c+d)(b+d)] /
	 * 							(n choose 2)^2 - [(a+b)(a+c) + (c+d)(b+d)]
	 */
	@Override
	public double evaluate() {
		// determine actual sorted clusters
    	same = 0;
    	diff = 0;
    	same1diff2 = 0;
    	diff1same2 = 0;
    	for (int i = 0; i < cluster1.length; i++) {
    		for (int j = 0; j < cluster1.length; j++) {
    			// if i and j in cluster 1 are in the same set and i and j in cluster 2 are in the same set 
    			if (cluster1[i] == cluster1[j] && cluster2[i] == cluster2[j]) {
    				same++;
    			} else if (cluster1[i] != cluster1[j] && cluster2[i] != cluster2[j]){
    				diff++;
    			} else if(cluster1[i] == cluster1[j] && cluster2[i] != cluster2[j]) {
    				same1diff2++;
    			} else if (cluster1[i] != cluster1[j] && cluster2[i] == cluster2[j]) {
    				diff1same2++;
    			}
    		}		
    	}
    	if (adjRandIndex == true) {
    		return getAdjustedRandIndex();
    	} else {
    		return getRandIndex();
    	}
	}
	
	/**
	 * The Rand Index, a + d / (N choose 2), where 
	 * a = number of pairs of elements that are in the same set in X and Y
	 * d = number of pairs of elements that are in different sets in X and 
	 * 		different sets in Y.
	 * 
	 * @return Rand Index
	 */
	private double getRandIndex() {
		return (same + diff) / ((1/2) * (cluster1.length - 1) * cluster1.length);
	}
	
	/**
	 * The Adjusted Rand Index, (n choose 2)(a + d) - [(a+b)(a+c) + (c+d)(b+d)] /
	 * 							(n choose 2)^2 - [(a+b)(a+c) + (c+d)(b+d)], where
	 * a = number of pairs of elements that are in the same set in X and Y
	 * b = number of pairs of elements that are in the same set in X and 
	 * 		different sets in Y
	 * c = number of pairs of elements that are in different sets in X and in the 
	 * 		same set in Y.
	 * d = number of pairs of elements that are in different sets in X and 
	 * 		different sets in Y.
	 *
	 * @return Adjusted Rand Index
	 */
	private double getAdjustedRandIndex() {
		double nchoose2 = (1/2) * (cluster1.length - 1) * cluster1.length;
		double sum = ((same + same1diff2) * (same + diff1same2) +
				(diff1same2 + diff) * (same1diff2 + diff));
		return (nchoose2 * (same + diff) - sum) / 
				(Math.pow(nchoose2, 2) - sum);
	}
	
	/**
	 * Constructor for DistinguishingPairs
	 */
	public DistinguishingPairs(int n, Instances d, int[] a, int[] b) {
		this.data = d;
		
		if (getNumClusters(a) != getNumClusters(b)) {
    		throw new IllegalArgumentException("Clusters cannot have"
    				+ "different number of total clusters");
		}
		this.cluster1 = a;
		this.cluster2 = b;
		
		if (n <= 0) {
    		throw new IllegalArgumentException("The number of times evaluated"
    				+ "cannot be fewer than 1.");
		}
		this.numEvals = n;
	}

}
