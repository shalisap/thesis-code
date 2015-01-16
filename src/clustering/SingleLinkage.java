package clustering;

/**
 * @author Shalisa Pattarawuttiwong
 * 
 * The Single-link method (also known as the nearest neighbor method, 
 * the minimum method, or the connectedness method) employs the nearest
 * neighbor distance to measure the dissimilarity between two groups.
 *
 */
public class SingleLinkage implements AgglomerationMethod {

	/** 
	 * The minimum of the dissimilarity between i and k, or between j and k is the 
	 * distance between two clusters.
	 * 
	 * distance between C(k) and C(i) U C(k) = 
	 * 		min{D(C(k), C(i), D(C(k), C(j))}, 
	 * 			where C(i), C(j), C(k) are groups of data points.
	 */
	@Override
	public double computeDist(double dik, double djk, double dij, 
			int ci, int cj, int ck) {
		return Math.min(dik, djk);
	}
	
	/**
	 * Returns the name of the agglomeration method being used.
	 */
	@Override
	public String toString() {
		return "Single-link method";
	}

}
