package clustering;

import weka.core.Instance;
import weka.core.Instances;
import SimilarityFn;

/**
 * Implementation of KMeans.
 *
 * @author Shalisa Pattarawuttiwong
 */
public class KMeans implements ClusterAlg {

    // also needs similarity fn
    SimilarityFn simFn;
    Instances data;
    int k = 2; // default value of k; number of clusters to generate

    void setK(int num) {
        k = num;
    }

    void cluster() {
    }

    int[][] getClusters(){
    }

   /**
    * Constructor for KMeans that takes data and
    * a similarity function.
    */
   public KMeans(Instances d, SimilarityFn s) {
        simFn = s;
        data = d;
   }
}
