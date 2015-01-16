package distance;

import weka.core.Instance;
import weka.core.Instances;

/**
 * Implementation of the Manhattan Distance.
 *
 * @author Shalisa Pattarawuttiwong
 */
public class ManhattanDistance extends AbstractDistance {

	Instance x;
	Instance y;
	Instances data;

	/**
	 * Calculates the distance between two instances.
	 *
	 * @return    the manhattan distance between the two instances
	 */
	@Override
	public double calculateDistance(Instance x, Instance y) {
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
	 * Calculates the dissimilarity matrix 
	 *
	 * @return    the dissimilarity matrix using the manhattan distance
	 */
	@Override
	public double[][] calculateDistMatrix(Instances data) {
		final double[][] disMatrix = new double[data.numInstances()][data.numInstances()];
		// fill diagonal
		for (int i = 0; i < disMatrix.length; i++) {
			disMatrix[i][i] = 0.0;
		}
		// Assuming symmetric measure, complete half and mirror across diagonal
		for (int i = 0; i < disMatrix.length; i++) {
			for (int j = 0; j < i; j++) {
				final double dist = calculateDistance(data.instance(i), data.instance(j));
				disMatrix[i][j] = dist;
				disMatrix[j][i] = dist;
			}
		}
		return disMatrix;
	}

	public ManhattanDistance(Instance a, Instance b) {
		this.x = a;
		this.y = b;
	}
	
	public ManhattanDistance(Instances x) {
		this.data = x;
	}
}
