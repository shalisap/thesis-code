package clustering;

/**
 * Interface for ClusterAlg.
 *
 * @author Shalisa Pattarawuttiwong
 */
public interface ClusterAlg {

	/**
	 * Given instances (data), D = [x{0}, x{1}, ..., x{n-1}], a 
	 * similarity/distance measure, and number of clusters
	 * to generate, k > 0, partitions the data such that there are 
	 * C{0}, ..., C{k-1} clusters of D.
	 */
    public void cluster();

    /**
     * Returns clusters [i{0}, ..., i{n-1}] where i{j} is a member of C{i},
     * where C{0}, ..., C{k-1} are partitions of D.
     * @return assignment of instances of the dataset to clusters.
     */
    public int[] getClusters();

}
