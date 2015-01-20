package clustering;

import java.util.Random;

import weka.core.Instance;
import weka.core.Instances;
import distance.DistanceFunction;

/**
 * Implementation of K-Medoids.
 * 
 * @author Shalisa Pattarawuttiwong
 */
public class KMedoids implements ClusterAlg {
	
    /**
     * Holds the similarity/distance function to be used
     */
    protected DistanceFunction distFn;

    /**
     * Holds the data to be processed
     */
    Instances data;

    /**
     * Holds the number of clusters to be generated
     */
    int numClusters = 2; // default value of k; number of clusters to generate

    /**
     * Holds the number of iterations
     */
    private int iterations = 1;

    /**
     * Holds the cluster centroids
     */
    private Instances centroids;

    /**
     * Holds the labels for each instance in the data
     */
    private int[] clusters;
    
    /**
     * Random number generator 
     */
    private Random rand;
    
    /**
     * Constructor for KMeans that takes data and
     * a similarity function.
     */
    public KMedoids(Instances d, DistanceFunction s)
 		   throws IllegalArgumentException {
         this.distFn = s;
         if (d.numInstances() <= 0) {
     		throw new IllegalArgumentException("The dataset"
     				+ " cannot be empty");
         } else this.data = d;
         rand = new Random(System.currentTimeMillis());
    }
    
    /**
     * Set the number of clusters to find
     * @param k Number of clusters
     */
    public void setNumClusters(int k)
    		throws IllegalArgumentException{
    	if (k <= 0) {
    		throw new IllegalArgumentException("Cannot set the number "
    				+ "of clusters to fewer than 1");
    	} else this.numClusters = k;
    }

    /**
     * Set the number of iterations to run
     * @param i Number of iterations
     */
    public void setNumIterations(int i)
    		throws IllegalArgumentException{
    	if (i <= 0) {
    		throw new IllegalArgumentException("Cannot set iterations "
    				+ "to fewer than 1");
    	} else this.iterations = i;
    }
    
	/** 
	 * 
	 */
	@Override
	public void cluster() {
		Instance[] medoids = new Instance[numClusters];
		clusters = new int[data.numInstances()];
		for (int i = 0; i < numClusters; i++) {
			int randNum = rand.nextInt(data.numInstances());
			// randomize first pick of medoids.
			medoids[i] = data.instance(randNum);
		}
		
		boolean changed = true;
		int count = 0;
		while (changed && count < iterations) {
			changed = false;
			count++;
			clusters = assign(medoids, data);
			changed = recalculateMedoids(clusters, medoids, data);
		}
	}

	/** 
	 * Returns the labels from KMedoids clustering of the data
	 */
	@Override
	public int[] getClusters() {
		if (this.clusters == null) {
			cluster();
		} 
		return this.clusters;
	}

}
