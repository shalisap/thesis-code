package distance;

import weka.core.Instance;

/**
 * Implementation of the Euclidean Distance.
 *
 * @author Shalisa Pattarawuttiwong
 */
public class EuclideanDistance extends AbstractDistance {

	/**
	 * Calculates the euclidean distance between two instances:
	 * d = square_root(sum over i = 1 to n ((x{i} - y{i})^2))),
	 * where n is the number of instances, x, y are instances, and 
	 * each x{i}, y{i} are sums of the values of 
	 * their attributes.
	 *
	 * @return the euclidean distance between the two instances
	 */
	@Override
	public double distance(Instance x, Instance y) {
		if (x.numAttributes() != y.numAttributes()) {
			throw new IllegalArgumentException("Both instances do not "
					+ "contain the same number of attributes");
		}
		double sum = 0.0;
		for (int i = 0; i < x.numAttributes(); i++){
			if (Double.isNaN(x.value(i)) || Double.isNaN(y.value(i))) {
				throw new IllegalArgumentException("One of the instances"
						+ "contains a NaN-valued attribute.");
			}
			
			sum += (y.value(i) - x.value(i)) * (y.value(i) - x.value(i)) ;
		}
		return Math.sqrt(sum);
	}

	/**
	 * Constructor for EuclideanDistance.
	 */
	public EuclideanDistance(){
	}
}
