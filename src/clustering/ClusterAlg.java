package clustering;

import weka.core.Instance;
import weka.core.Instances;

/**
 * Interface for ClusterAlg.
 *
 * @author Shalisa Pattarawuttiwong
 */
public interface ClusterAlg {

     void cluster();

     int[][] getClusters();

}
