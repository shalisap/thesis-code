package distance;

import java.io.Serializable;
/**
 * Interface for Similarity/Distance Functions
 *
 * @author Shalisa Pattarawuttiwong
 */
public interface DistanceFunction extends Serializable{

     public double calculateDistance();

}
