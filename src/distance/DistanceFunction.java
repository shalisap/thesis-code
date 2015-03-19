package distance;

import java.io.Serializable;
import weka.core.Instance;
import weka.core.Instances;

/**
 * Interface for Similarity/Distance Functions
 *
 * @author Shalisa Pattarawuttiwong
 */
public interface DistanceFunction extends Serializable {

	/**
	 * The distance (double) between x and y, where x and y are 
	 *  Instances 
	 * 
	 * @param x Instance
	 * @param y Instance
	 * @return distance between x and y
	 */
	public double distance(Instance x, Instance y);
	
	/**
	 * The distance matrix of the data provided.
	 * 
	 * @param data
	 * @return distance matrix of the data given
	 */
	public double[][] distMatrix(Instances data);

}
