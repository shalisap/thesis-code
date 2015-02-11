package distance;

import weka.core.Instance;
import weka.core.Instances;

/**
 * Implementation of the Euclidean Distance.
 *
 * @author Shalisa Pattarawuttiwong
 */
public class EuclideanDistance extends AbstractDistance {

	/**
	 * The first instance in order to calculate distance
	 */
	Instance x;
	
	/**
	 * The second instance in order to calculate distance
	 */
	Instance y;
	
	/**
	 * The data whose distance matrix is computed 
	 */
	Instances data;

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
			if (!Double.isNaN(x.value(i)) && !Double.isNaN(y.value(i))) {
				sum += Math.pow((y.value(i) - x.value(i)), 2);
			}
		}
		return Math.sqrt(sum);
	}

	/**
	 * Constructor for EuclideanDistance that takes in two separate
	 * instances, mainly for calculating the distance between the two.
	 * 
	 * @param a Instance
	 * @param b Instance
	 */
	public EuclideanDistance(Instance a, Instance b) {
		this.x = a;
		this.y = b;
	}
	
	/**
	 * Constructor for EuclideanDistance that takes in Instances, 
	 * mainly for constructing a distance matrix.
	 * 
	 * @param x Instances
	 */
	public EuclideanDistance(Instances x) {
		this.data = x;
	}

	/**
	 * Constructor for EuclideanDistance that allows it to 
	 * be passed as a function.
	 */
	public EuclideanDistance(){
	}
}
