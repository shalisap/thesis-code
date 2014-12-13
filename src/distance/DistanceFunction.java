package distance;

import java.io.Serializable;
import weka.core.Instance;

/**
 * Interface for Similarity/Distance Functions
 * 
 * @author Shalisa Pattarawuttiwong
 */
public interface DistanceFunction extends Serializable {
	
	public double calculateDistance(Instance x, Instance y);
	
	public boolean compare(double a, double b);
}
