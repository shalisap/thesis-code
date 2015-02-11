package distance;

import weka.core.Instance;
import weka.core.Instances;

/**
 * Implementation of the Manhattan Distance.
 *
 * @author Shalisa Pattarawuttiwong
 */
public class ManhattanDistance extends AbstractDistance {

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
	 * Calculates the manhattan distance between two instances:
	 * d = sum over i = 1 to n (|x{i} - y{i}|) where n is the 
	 * number of instances, and x, y are instances, and 
	 * each x{i}, y{i} are sums of the values of 
	 * their attributes.
	 *
	 * @return the manhattan distance between the two instances
	 */
	@Override
	public double distance(Instance x, Instance y) {
		if (x.numAttributes() != y.numAttributes()) {
			throw new IllegalArgumentException("Both instances do not "
					+ "contain the same number of attributes");
		}
		double sum = 0.0;
		for (int i = 0; i < x.numAttributes(); i++) {
			sum += Math.abs(x.value(i) - y.value(i));
		}
		return sum;
	}

	/**
	 * Constructor for ManhattanDistance that takes in two separate
	 * instances, mainly for calculating the distance between the two.
	 * 
	 * @param a Instance
	 * @param b Instance
	 */
	public ManhattanDistance(Instance a, Instance b) {
		this.x = a;
		this.y = b;
	}
	
	/**
	 * Constructor for ManhattanDistance that takes in Instances, 
	 * mainly for constructing a distance matrix.
	 * 
	 * @param x Instances whose distance matrix will be computed
	 */
	public ManhattanDistance(Instances x) {
		this.data = x;
	}
	
	/**
	 * Constructor for ManhattanDistance that allows it to 
	 * be passed as a function.
	 */
	public ManhattanDistance() {
	}
}
