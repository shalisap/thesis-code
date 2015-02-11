package distance;

import weka.core.Instances;

/**
 * Abstract super class for all distance measures. High values denote more
 * distant instances.
 *
 * @author Shalisa Pattarawuttiwong
 */
public abstract class AbstractDistance implements DistanceFunction {

	/**
	 * Calculates the distance matrix containing the distances
	 * of a set of instances (data).
	 *
	 * @return the distance matrix using the distance function
	 */
	@Override
	public double[][] distMatrix(Instances data) {
		final double[][] disMatrix = new double[data.numInstances()][data.numInstances()];
		// fill diagonal
		for (int i = 0; i < disMatrix.length; i++) {
			disMatrix[i][i] = 0.0;
		}
		// Assuming symmetric measure, complete half and mirror across diagonal
		for (int i = 0; i < disMatrix.length; i++) {
			for (int j = 0; j < i; j++) {
				final double dist = distance(data.instance(i), data.instance(j));
				disMatrix[i][j] = dist;
				disMatrix[j][i] = dist;
			}
		}
		return disMatrix;
	}

}
