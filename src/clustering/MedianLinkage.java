package clustering;

/**
 * @author Shalisa Pattarawuttiwong
 *
 * The median method, also known as the weighted centroid method, 
 * is similar to the centroid method except the sizes of the clusters
 * is assumed to be equal and the position of the new centroid
 * is always between two old centroids. It should mainly be used
 * only for the Euclidean distance.
 */
public class MedianLinkage implements AgglomerationMethod {

	/** 
	 * The distance between two clusters is the distance between their
	 * weighted centroids =
	 * 		0.5 * D(C(k), C(i)) + 0.5 * D(C(k), C(j)) - 0.25 * (D(C(i), C(j))
	 * 		where C(i), C(j), C(k) are groups of data points.
	 */
	@Override
	public double computeDist(double dik, double djk, double dij, int numi,
			int numj, int numk) {
		return 0.5 * dik + 0.5 * djk - 0.25 * dij;
	}

	/**
	 * @return name of the agglomeration method chosen
	 */
	public String toString() {
		return "Median method";
	}
}
