/**
 * 
 */
package clustering;

/**
 * Interface for using the various 
 * agglomeration methods.
 * 
 * @author Shalisa Pattarawuttiwong
 *
 */
public interface AgglomerationMethod {

	/**
	 * Compute the dissimilarity between the 
	 * newly formed cluster (i,j) and the existing cluster k.
	 * 
	 * @param dik dissimilarity between clusters i and k
	 * @param djk dissimilarity between clusters j and k
	 * @param dij dissimilarity between clusters i and j
	 * @param numi number of elements in cluster i
	 * @param numj number of elements cluster j
	 * @param numk number of elements cluster k
	 * 
	 * @return dissimilarity between cluster (i,j) and cluster k.
	 */
	public double computeDist(double dik, double djk, double dij, 
			int numi, int numj, int numk);

	/**
	 * 
	 * @return name of the agglomeration method chosen
	 */
	public String toString();
}
