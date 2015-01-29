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
     * Determines the actual values in each cluster. 
     * @param clusters
     * @param 
     */
    public ArrayList<ArrayList<String>> determineClusters(int[] clusters) {
    	ArrayList<ArrayList<String>> actualClusters = new ArrayList<ArrayList<String>>();
    	HashMap<Integer, ArrayList<String>> clusterValues = new HashMap<Integer, ArrayList<String>>();
    	int numInst = 0;
    	for (int i : clusters) {
    		if (!clusterValues.containsKey(i)) {
    			ArrayList<String> c = new ArrayList<String>();
    			c.add(data.instance(numInst).toString());
    			clusterValues.put(i, c);
    		} else {
    			ArrayList<String> c = clusterValues.get(i);
    			c.add(data.instance(numInst).toString());
    			clusterValues.put(i, c);
    		}
    		numInst++;
    	}
    	
    	for (int k: clusterValues.keySet()) {
			ArrayList<String> c = clusterValues.get(k);
    		Collections.sort(c);
    		actualClusters.add(c);
    	}
    	
    	Collections.sort(actualClusters, new Comparator<ArrayList<String>>() {
    		public int compare(ArrayList<String> a, ArrayList<String> b) {
    			return a.get(0).compareTo(b.get(0));
    		}
    	});
    	
    	return actualClusters;
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
	 * b = number of pairs of elements in N that are in different sets in X and 
	 * 		different sets in Y.
	 * c = number of pairs of elements in N that are in the same set in X and 
	 * 		different sets in Y
	 * d = number of pairs of elements in N that are in different sets in X and in the 
	 * 		same set in Y.
	 * 
	 * c(Y, Y') = a + b / a + b + c + d = a + b / (N choose 2)
	 * 
	 * c = 0 when two clusterings have no similarities -> 1 where they're identical.
	 */
	@Override
	public double evaluate() {
		// determine actual sorted clusters
		ArrayList<ArrayList<String>> actualCluster1 = determineClusters(cluster1);
		ArrayList<ArrayList<String>> actualCluster2 = determineClusters(cluster2);
    	int same = 0;
    	int diff = 0;
    	for (int i = 0; i < actualCluster1.size(); i++) {
    		for (int j = 0; j < actualCluster1.get(i).size(); j++) {
    			// cannot directly compare...
    			if (actualCluster1.get(i).get(j) == actualCluster2.get(i).get(j)) {
    				
    			}
    		}
    	}
		return 0.0;

	}
	
	public DistinguishingPairs(int n,Instances d, int[] a, int[] b) {
		this.data = d;
		
		if (getNumClusters(cluster1) != getNumClusters(cluster2))
    		throw new IllegalArgumentException("Clusters cannot have"
    				+ "different number of total clusters");
		this.cluster1 = a;
		this.cluster2 = b;

		if (n <= 0) {
    		throw new IllegalArgumentException("The number of times evaluated"
    				+ "cannot be fewer than 1.");
		}
		this.numEvals = n;
	}

}
