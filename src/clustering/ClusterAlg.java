package clustering;

/**
 * Interface for ClusterAlg.
 *
 * @author Shalisa Pattarawuttiwong
 */
public interface ClusterAlg {

	/**
	 * Given instances, [x0, x1, ..., xn], and number of clusters, y > 0,
	 * assigns a cluster number 0...y to the instance, such that
	 * each instance maps to a cluster 
	 * (e.g. [(0...y)0, (0...y)1,...,(0...y)n] of size n).
	 */
    public void cluster();

    /**
     * 
     * @return assignment of instances to clusters, 
     * 		[(0...y)0, (0...y)1,..., (0...y)n] where 
     * 		it is mapped to given instances [x0, x1, ..., xn] and
     * 		each 0...y is the cluster number the instance is assigned to and . 
     */
    public int[] getClusters();

}
