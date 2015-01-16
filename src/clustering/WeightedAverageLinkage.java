package clustering;

/**
 * @author Shalisa Pattarawuttiwong
 *
 * The weighted group average method is similar to the group average
 * method except that the sizes of the clusters are assumed to be equal.
 *
 */
public class WeightedAverageLinkage implements AgglomerationMethod {

	/** 
	 * The weighted average of the distances between all pairs of objects
	 * in opposite clusters.
	 * 
	 * weighted average of distances between C(k) and C(i) U C(j) = 
	 * 		0.5 * D(C(k), C(i)) + 0.5 * D(C(k), C(j))
	 * 		where C(i), C(j), C(k) are groups of data points,
	 */
	@Override
	public double computeDist(double dik, double djk, double dij, int numi,
			int numj, int numk) {
		return 0.5 * dik + 0.5 * djk;
	}

	/**
	 * Returns the name of the agglomeration method being used.
	 */
	@Override
	public String toString() {
		return "Weighted group average method";
	}
	
}
