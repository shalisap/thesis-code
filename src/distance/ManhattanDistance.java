package distance;

import weka.core.Instance;
/**
 * Implementation of the Manhattan Distance.
 *
 * @author Shalisa Pattarawuttiwong
 */
public class ManhattanDistance extends AbstractDistance {

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
			// charge 1 for -1
//			double xval;
//			double yval;
//			if (y.value(i) == -1.0) {
//				yval = 1.0;
//			} else {
//				yval = y.value(i);
//			}
//			if (x.value(i) == -1.0) {
//				xval = 1.0;
//			} else {
//				xval = x.value(i);
//			}
			
			sum += Math.abs(x.value(i) - y.value(i));
		}
		return sum;
	}
	
	/**
	 * Constructor for ManhattanDistance.
	 */
	public ManhattanDistance() {
	}
}
