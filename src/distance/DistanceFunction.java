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

	public double calculateDistance(Instance x, Instance y);
	
	public double[][] calculateDistMatrix(Instances data);

	public boolean compare(double a, double b);

}
