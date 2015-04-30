package clustering;

/**
 * @author Shalisa Pattarawuttiwong
 *
 * The centroid method links the centroids of clusters, where
 * the distance between two clusters is calculated as the 
 * distance between their centroids. Should generally be used
 * only for the Euclidean distance.
 */
public class CentroidLinkage implements AgglomerationMethod {

	/** 
	 * The distance between two clusters is the distance between their
	 * cluster centroids following the Lance-Williams formula =
	 *		(|C(i)| / |C(i)| + |C(j)|) * D(C(k), C(i)) +
	 *		(|C(j)| / |C(i)| + |C(j)|) * D(C(k), C(j)) -
	 *		(|C(i)|*|C(j)| / (|C(i)| + |C(j)|)^2) * D(C(i), C(j))  
	 *		where C(i), C(j), C(k) are groups of data points,
	 */
	@Override
	public double computeDist(double dik, double djk, double dij, int numi,
			int numj, int numk) {
		return (numi * dik + numj * djk - numi * numj * dij 
				/ (numi + numj)) / (numi + numj);
	}
	
	/**
	 * @return name of the agglomeration method chosen
	 */
	public String toString() {
		return "Centroid method";
	}

}
