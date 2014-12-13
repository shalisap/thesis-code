package distance;

import weka.core.Instance;
import weka.core.Instances;

/**
 * Implementation of the Euclidean Distance.
 * 
 * @author Shalisa Pattarawuttiwong
 */
public class EuclideanDistance extends AbstractDistance {

	Instance x;
	Instance y;
	
	/**
	 * Calculates the distance between two instances.
	 * 
	 * @return    the euclidean distance between the two instances
	 */
	@Override
	public double calculateDistance(Instance x, Instance y) {
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
	
	public EuclideanDistance(Instance a, Instance b) {
		x = a;
		y = b;
	}
	
	public EuclideanDistance(){
		
	}
}
