package clustering;

/**
 * @author Shalisa Pattarawuttiwong
 *
 * Ward's method, also known as the minimum variance method,
 * forms partitions in a manner that minimizes the loss of 
 * information associated with each merging.
 */
public class WardLinkage implements AgglomerationMethod {

	/** 
	 *  The distances between C(k) and C(i) U C(j) = 
	 * 		|C(k)| + |C(i)| / sumijk * D(C(k), C(i)) +
	 * 		|C(k)| + |C(j)| / sumijk * D(C(k), C(j)) -
	 * 		|C(k)| / sumijk * D(C(i),C(j))
	 * 		where C(i), C(j), C(k) are groups of data points, 
	 * 		and sumijk = |C(k)| + |C(i)| + |C(j)|
	 */
	@Override
	public double computeDist(double dik, double djk, double dij, int numi,
			int numj, int numk) {
		return ((numi + numk) * dik + (numj + numk) * djk - numk * dij) / (numi + numj + numk);
	}
	
	/**
	 * Returns the name of the agglomeration method being used.
	 */
	@Override
	public String toString() {
		return "Ward's method";
	}

}
