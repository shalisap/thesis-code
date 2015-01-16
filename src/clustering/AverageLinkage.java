package clustering;

/**
 * @author Shalisa Pattarawuttiwong
 * 
 * The group average method, also known as unweighted pair group method
 * using arithmetic averages (UPGMA), uses the averages of the distances
 * between all possible pairs of data points. 
 *
 */
public class AverageLinkage implements AgglomerationMethod {

	/** 
	 * The average of the distances between all pairs of objects
	 * in opposite clusters.
	 * 
	 * average of distances between C(k) and C(i) U C(j) = 
	 * 		1/(|C(k)||C(i) U C(j)) * sum (D(x,y))
	 * 		where C(i), C(j), C(k) are groups of data points,
	 * 		and x is a data point from C(k), y is a data point from C(i) U C(j).
	 */
	@Override
	public double computeDist(double dik, double djk, double dij, int numi,
			int numj, int numk) {
		return (numi * dik + numj * djk) / (numi + numj);
	}
	
	/**
	 * Returns the name of the agglomeration method being used.
	 */
	@Override
	public String toString() {
		return "Group average method (UPGMA)";
	}

}
