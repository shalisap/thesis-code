package clustering;

/**
 * @author Shalisa Pattarawuttiwong
 * 
 * The complete link method employs the farthest neighbor distance to 
 * measure the dissimilarity between two groups.
 *
 */
public class CompleteLinkage implements AgglomerationMethod {

	/** 
	 * The maximum of the dissimilarity between i and k, or between j and k is the 
	 * distance between two clusters.
	 * 
	 * distance between C(k) and C(i) U C(k) = 
	 * 		max{D(C(k), C(i), D(C(k), C(j))}, 
	 * 			where C(i), C(j), C(k) are groups of data points.
	 */
	@Override
	public double computeDist(double dik, double djk, double dij, int ci,
			int cj, int ck) {
		return Math.max(dik, djk);
	}
	
	/**
	 * Returns the name of the agglomeration method being used.
	 */
	@Override
	public String toString() {
		return "Complete link method";
	}

}
