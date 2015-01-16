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
	Instances data;

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
	
	/**
	 * Calculates the dissimilarity matrix 
	 *
	 * @return    the dissimilarity matrix using the euclidean distance
	 */
	@Override
	public double[][] calculateDistMatrix(Instances data) {
		double[][] disMatrix = new double[data.numInstances()][data.numInstances()];
		// fill diagonal
		for (int i = 0; i < disMatrix.length; i++) {
			disMatrix[i][i] = 0.0;
		}
		// Assuming symmetric measure, complete half and mirror across diagonal
		for (int i = 0; i < disMatrix.length; i++) {
			for (int j = 0; j < i; j++) {
				double dist = calculateDistance(data.instance(i), data.instance(j));
				disMatrix[i][j] = dist;
				disMatrix[j][i] = dist;
			}
		}
		return disMatrix;
	}

	public EuclideanDistance(Instance a, Instance b) {
		this.x = a;
		this.y = b;
	}
	
	public EuclideanDistance(Instances x) {
		this.data = x;
	}

	public EuclideanDistance(){

	}
}
