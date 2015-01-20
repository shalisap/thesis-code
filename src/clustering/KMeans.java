package clustering;

import java.util.*;

import weka.core.Instances;
import distance.DistanceFunction;
import weka.core.Attribute;
import weka.core.Instance;

/**
 * Implementation of K-Means.
 *
 * @author Shalisa Pattarawuttiwong
 */
public class KMeans implements ClusterAlg {

    /**
     * Holds the similarity/distance function to be used
     */
    protected DistanceFunction distFn;

    /**
     * Holds the data to be processed
     */
    Instances data;

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
     * If true, allows the user to pick the initial centroids,
     * if false, randomize the initial centroids.
     */
    private boolean chooseInitCentroids = false;
    	
    /**
     * Constructor for KMeans that takes data and
     * a similarity function.
     */
    public KMeans(Instances d, DistanceFunction s)
 		   throws IllegalArgumentException {
         this.distFn = s;
         if (d.numInstances() <= 0) {
     		throw new IllegalArgumentException("The dataset"
     				+ " cannot be empty");
         } else this.data = d;
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
     * Choose to randomize or manually pick centroids.
     * @param i True for user's pick, false for randomize.
     */
    public void setChooseInitCentroids(boolean i) {
    	this.chooseInitCentroids = i;
    }

    /**
     * Randomizes the initial centroids chosen.
     */
    private void randomizeInitCentroids(){
        Random rand = new Random(); // random number generator
        while (this.centroids.numInstances() < this.numClusters) {
        	boolean addRandom = true;
        	Instance randomInstance = this.data.instance(
        			rand.nextInt(this.data.numInstances()));
        	for (int k = 0; k < this.centroids.numInstances(); k++) {
        		if (randomInstance == this.centroids.instance(k)) {
        			addRandom = false;
        		}
        	}
        	if (addRandom == true) {
        		this.centroids.add(randomInstance);
        	}
        }
    }
    
    /**
     * Allows the user to choose the initial centroids.
     */
    public void pickInitCentroids(){
    	// print instances
    	boolean duplicate = true;
    	ArrayList<String> indices = new ArrayList<String>();
    	
    	System.out.println("Instances");
    	for (int i = 0; i < this.data.numInstances(); i++) {
    		System.out.println(i + ": " + this.data.instance(i));
    	}
    	
		System.out.println("Enter indices of the distinct centroids wanted (Ex: 0 1 done): ");
    	Scanner input = new Scanner(System.in);
    	
    	String index = input.next();
    	while ((!index.equals("done")) && duplicate == true) {
    		if (indices.contains(index)) {
    			duplicate = false;
    		}
    		else {
    			indices.add(index);
	    		Instance chosenInstance = this.data.instance(Integer.parseInt(index));	
	    		this.centroids.add(chosenInstance);
	    		index = input.next();
    		}
    	}
		System.out.println();
    }
    
    /**
     * Runs the kmeans clustering algorithm.
     */
	@Override
	public void cluster() {
        Random rand = new Random(); // random number generator
        this.centroids = new Instances(this.data, this.numClusters);
        int instanceLength = this.data.instance(0).numAttributes();

        // Create instances that contain the min/max values for the attributes 
        // -- move to own function?
        Instance min = new Instance(instanceLength);
        Instance max = new Instance(instanceLength);
        // for each instance -- not iterable
        for (int i = 0; i < this.data.numInstances(); i++) {
        	Instance inst = this.data.instance(i);
        	// for each attribute
        	for (int j = 0; j < inst.numAttributes(); j++) {
        		Attribute att = inst.attribute(j);
        		double val = inst.value(j);
        		if (max.isMissing(att) && min.isMissing(att)) {
        			max.setValue(j, val);
        			min.setValue(j, val);
        		} else if (max.value(j) < val) {
        			max.setValue(j, val);
        		} else if (min.value(j) > val) {
        			min.setValue(j, val);
        		}
        	}
        }

        if (this.chooseInitCentroids == false) {
        	randomizeInitCentroids();
        } else {
        	pickInitCentroids();
        	while (this.centroids.numInstances() != this.numClusters) {
        		System.out.println("Centroids chosen not equal to "
        				+ "the number of clusters wanted or "
        				+ "duplicates of centroids chosen");
                this.centroids = new Instances(this.data, this.numClusters);
        		pickInitCentroids();
        	}
        }

        int iterationCount = 0;
        this.clusters = new int[this.data.numInstances()];
		// assign each object to the group with the closest centroid
		for (int i = 0; i < this.data.numInstances(); i++) {
			int tmpCluster = 0;
			double minDistance = distFn.calculateDistance(
					this.centroids.instance(0), this.data.instance(i));
			for (int j = 1; j < this.centroids.numInstances(); j++) {
				double dist = distFn.calculateDistance(
						this.centroids.instance(j),
						this.data.instance(i));
				if (distFn.compare(dist, minDistance)) {
					minDistance = dist;
					tmpCluster = j;
				}
			}
			this.clusters[i] = tmpCluster;
		}
		iterationCount++;
        
        boolean centroidsChanged = true;
        boolean randomCentroids = true;
        while (randomCentroids || (iterationCount < this.iterations
        		&& centroidsChanged)) {
        	iterationCount++;

            // When all objects assigned, recalculate positions of K
        	// centroids, start over. The new centroid is the weighted
        	// center of the current cluster
        	double[][] sumPosition = new
        			double[this.numClusters][instanceLength];
        	int[] countPosition = new int[this.numClusters];
        	for (int i = 0; i < this.data.numInstances(); i++) {
        		Instance in = this.data.instance(i);
        		for (int j = 0; j < instanceLength; j++) {
        			sumPosition[this.clusters[i]][j] += in.value(j);
        		}
        		countPosition[this.clusters[i]]++;
        	}
        	centroidsChanged = false;
        	randomCentroids = false;

        	for (int i = 0; i < this.numClusters; i++) {
        		if (countPosition[i] > 0) {
        			Instance newCentroid = new Instance(instanceLength);
        			for (int j = 0; j < instanceLength; j++) {
        				newCentroid.setValue(j,
        						(float) sumPosition[i][j] / countPosition[i]);
        			}
        			if (distFn.calculateDistance(newCentroid,
        					this.centroids.instance(i)) > 0.0001) {
        				centroidsChanged = true;
        				// write a replace method in Instance.java?
        				this.centroids.delete(i);
        				this.centroids.add(newCentroid);
        			}
        		} else {
        			Instance randomInstance = new Instance(instanceLength);
        			for (int j = 0; j < instanceLength; j++) {
        				double dist = Math.abs(max.value(j) - min.value(j));
        				randomInstance.setValue(j, (float)
        						(min.value(j) + rand.nextDouble() * dist));
        			}
        			randomCentroids = true;
        			// replace?
        			this.centroids.delete(i);
        			this.centroids.add(randomInstance);
        		}
        	}

        	this.clusters = new int[this.data.numInstances()];
        	for (int i = 0; i < this.data.numInstances(); i++) {
        		int tmpCluster = 0;
        		double minDistance = distFn.calculateDistance(
        				centroids.instance(0), this.data.instance(i));
        		for (int j = 0; j < this.centroids.numInstances(); j++) {
        			double dist = distFn.calculateDistance(
        					this.centroids.instance(j), this.data.instance(i));
        			if (distFn.compare(dist, minDistance)) {
        				minDistance = dist;
        				tmpCluster = j;
        			}
        		}
        		this.clusters[i] = tmpCluster;
        	}
        }
	}

	/**
	 * Returns the labels from KMeans clustering of the data.
	 */
	@Override
	public int[] getClusters() {
		if (this.clusters == null) {
			cluster();
		} 
		return this.clusters;
	}
}
