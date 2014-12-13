package clustering;

import java.util.Random;
import java.util.Arrays;

import weka.core.Instance;
import weka.core.Instances;
import weka.core.Attribute;
import distance.DistanceFunction;

/**
 * Implementation of Hierarchical Agglomerative Clustering
 *
 * @author Shalisa Pattarawuttiwong
 */
public class HierAgglo implements ClusterAlg {

    /**
     * Holds the similarity/distance function to be used
     * single linkage: distance of 2 closest objects in diff clusters
     * complete linkage: greatest distance between obj in diff clusters
     * group average linkage: average dist between all pairs in obj in diff clusters
     */
    protected DistanceFunction distFn;

    /**
     * Holds the data to be processed
     */
    Instances data;

    int numClusters = 2; // default value of k; number of clusters to generate

    /**
     * Holds the labels for each instance in the data
     */
    private int[] output;

    /**
     * Set the number of clusters to find
     * @param k Number of clusters
     */
    public void setNumClusters(int k) {
         this.numClusters = k;
    }

    /**
     * Runs the hierarchical agglomerative clustering algorithm.
     * Implementation similar to the implementation of kmeans in the Java Machine Learning Library.
     */

    @Override
    public void cluster() {
    }

    /**
     * Returns the labels from hierarchical agglomerative clustering of the data
     */
    @Override
    public int[] getClusters(){
        return this.output;
    }

   /**
    * Constructor for HierAgglo that takes data and
    * a similarity function.
    */
   public HierAgglo(Instances d, DistanceFunction s) {
        this.distFn = s;
        this.data = d;
   }
}
