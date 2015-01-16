/**
 * 
 */
package clustering;

/**
 * @author Shalisa Pattarawuttiwong
 *
 */
public class SingleLinkage implements AgglomerationMethod {

	/* 
	 * Employs the nearest neighbor distance to measure the dissimilariy 
	 * between two groups.
	 * The minimum of the dissimilarity between i and k, or between j and k are taken
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
