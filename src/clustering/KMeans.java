package clustering;

import java.util.*;

import weka.core.Instances;
import distance.DistanceFunction;
import weka.core.Instance;

/**
 * Implementation of K-Means clustering.
 *
 * @author Shalisa Pattarawuttiwong
 */
public class KMeans implements ClusterAlg {

    /**
     * The similarity/distance function to be used
     */
    protected DistanceFunction distFn;

    /**
     * The data to be clustered
     */
    protected Instances data;

    /**
     * The number of clusters, k, to generate
     */
    protected int numClusters = 2; // default value of k

    /**
     * The number of maximum iterations to run cluster()
     */
    private int iterations = 1;

    /**
     * The cluster centroids, [c{0}, c{1}, ..., c{n-1}], where
     * each c{i}, 0 <= i < n, is an instance representing 
     * a centroid and n is the number of clusters.
     */
    private Instance[] centroids;

    /**
     * [i{0}, i{1}, ..., i{n-1}] where each i{j}, 0 <= j < n belongs 
     * to a partition, C{k}, 0 <= k <= numClusters, of the dataset. 
     */
    private int[] clusters;
    
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
//         } else if (d.numAttributes() % 2 != 0) {
//      		throw new IllegalArgumentException("The dataset"
//      				+ " has an odd number of attributes. It must"
//      				+ " have pairs of (IN, OUT).");
         } else this.data = d;
    }
    
    /**
     * Set the number of clusters to generate
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
     * Set the number of maximum iterations to run cluster()
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
     * Randomizes the initial centroids chosen.
     */
    private void randomizeInitCentroids(){
        centroids = new Instance[this.numClusters];
        
        // construct list xs where xs[i] = i
        List<Integer> random = new ArrayList<Integer>();
        for (int i = 0; i < data.numInstances(); i++) {
        	random.add(i);
        }
        
        // shuffle xs
        Collections.shuffle(random);
        
        for (int k = 0; k < numClusters; k++) {
        	centroids[k] = data.instance(random.get(k));
        }
        
    }
    
    /**
     * Return the nearest instance in the dataset from a given instance.
     * @param d Instances dataset
     * @param inst Instance to find the closest other instance
     * @return Nearest instance to inst
     */
    private Map<Instance, Double> getNearestCent(Instances d, Instance inst) {
    	Map<Instance, Double> nearDist = new HashMap<Instance, Double>();
    	double minDistance = Double.MAX_VALUE;
    	Instance minCenter = null;
    	for (int i = 0; i < d.numInstances(); i++) {
    		double dist = distFn.distance(inst, d.instance(i));
    		if (dist < minDistance) {
    			minDistance = dist;
    			minCenter = d.instance(i);
    		}
    	}
    	nearDist.put(minCenter, minDistance);
    	return nearDist;
    }
    
    /**
     * k-means++ to choose initial centroids.
     */
    private void kMeansPlusPlusInit(){
    	centroids = new Instance[this.numClusters];
    	boolean[] taken = new boolean[data.numInstances()];
    	
    	// choose one center at random
    	Random rand = new Random();
    	int idx = rand.nextInt(data.numInstances());
    	Instance center = data.instance(idx);
    	this.centroids[0] = center;
    	// mark index as taken
    	taken[idx] = true;
    	
    	// keep track of min distance squared of elements of data
    	// to elements of centroids
    	double[] minDistSquared = new double[data.numInstances()];
    	for (int i = 0; i < data.numInstances(); i++) {
    		if (i != idx) {
    			// distance from first center to all others
    			double d = distFn.distance(center, data.instance(i));
    			minDistSquared[i] = d*d; 
    		}
    	}
    	
    	int chosen = 1;
    	while(chosen < this.numClusters) {
    		// Sum up distances for points not already taken.
    		double sqSum = 0.0;
    		for (int i = 0; i < data.numInstances(); i++) {
    			if (!taken[i]) {
    				sqSum += minDistSquared[i];
    			}
    		}
	    	// add random point chosen with probability
	    	// proportional to D(x)^2
    		// sum through minDistSquared until sum >= r
    		double r = rand.nextDouble() * sqSum;
    		int nextIdx = -1;
    		double sum = 0.0;
    		for (int i = 0; i < data.numInstances(); i++) {
    			if (!taken[i]) {
    				sum += minDistSquared[i];
    				if (sum >= r) {
    					nextIdx = i;
    					break;
    				}
    			}
    		}
    		
    		// check to make sure point was chosen
    		if (nextIdx == -1) {
    			for (int i = data.numInstances() - 1; i >= 0; i--) {
    				if (!taken[i]) {
    					nextIdx = i;
    					break;
    				}
    			}
    		}
    		
    		if (nextIdx >= 0) {
    			Instance nextCenter = data.instance(nextIdx);
    			this.centroids[chosen] = nextCenter;
    			taken[nextIdx] = true;
    			
    			// update minDistSquared
    			if (chosen + 1 < this.numClusters) {
    				for (int j = 0; j < data.numInstances(); j++) {
    					if (!taken[j]) {
    						double dist = distFn.distance(nextCenter, data.instance(j));
    						double distSq = dist*dist;
    						if (distSq < minDistSquared[j]) {
    							minDistSquared[j] = distSq;
    						}
    					}
    				}
    			} 
    		} else {
				// No next idx - exit
				break;
			}
			chosen++;
    	}

    }
    
    /**
     * Allows the user to choose the initial centroids.
     * @param pickedCent Set of integers representing indices of the data
     */
    public void setInitCentroids(Set<Integer> pickedCent){
    	
        centroids = new Instance[this.numClusters];
        int index = 0;
    	for (int item: pickedCent) {
    		centroids[index] = data.instance(item);
    		//centroids.add(this.data.instance(item));
    		index++;
    	}
  
    	if (centroids.length != numClusters) {
    		throw new IllegalArgumentException("Centroids chosen not equal to "
    				+ "the number of clusters wanted or "
    				+ "duplicates of centroids chosen");
    	}
    }
    
	/**
	 * Returns the clusters from KMeans clustering of the data, 
     * [i{0}, i{1}, ..., i{n-1}] where each i{j}, 0 <= j < n belongs 
     * to a partition, C{k}, 0 <= k <= numClusters, of the dataset. 
	 * 
	 */
	@Override
	public int[] getClusters() {
		if (this.clusters == null) {
			cluster();
		} 
		return this.clusters;
	}
	
    /**
     * Runs the kmeans clustering algorithm on the data given.
     */
	@Override
	public void cluster() {
        int instanceLength = this.data.instance(0).numAttributes();

        if (this.centroids == null) {
        	//randomizeInitCentroids();
        	kMeansPlusPlusInit();
        	System.out.println(Arrays.toString(this.centroids));
        }
        
        int iterationCount = 0;
        this.clusters = new int[this.data.numInstances()];
		// assign each object to the group with the closest centroid
		for (int i = 0; i < this.data.numInstances(); i++) {
			int tmpCluster = 0;
			double minDistance = distFn.distance(
					centroids[0], data.instance(i)); 
					//this.centroids.instance(0), this.data.instance(i));
			for (int j = 1; j < centroids.length; j++) {
				double dist = distFn.distance(
						centroids[j], data.instance(i));
						//this.centroids.instance(j),
						//this.data.instance(i));
				if (dist < minDistance) {
					minDistance = dist;
					tmpCluster = j;
				}
			}
			this.clusters[i] = tmpCluster;
		}
		iterationCount++;
        
        boolean centroidsChanged = true;
        
        /* While the number of iterations ran is fewer than 
         * the maximum set number of iterations, and 
         * while the centroids are still changing, recalculate centroids and clusters
         */
        while (iterationCount < iterations && centroidsChanged) {
        	iterationCount++;

            /* When all objects assigned, recalculate positions of the
        	 * centroids, start over. The new centroid is the weighted
        	 * center of the current cluster
        	 */
        	// sum of the values of instances in each cluster
        	double[][] sumPosition = new
        			double[numClusters][instanceLength];
        	// number of instances in each cluster
        	int[] countPosition = new int[numClusters];
        	for (int i = 0; i < data.numInstances(); i++) {
        		Instance in = data.instance(i);
        		for (int j = 0; j < instanceLength; j++) {
        			sumPosition[clusters[i]][j] += in.value(j);
        		}
        		countPosition[clusters[i]]++;
        	}
        	centroidsChanged = false;
        	
        	// Recalculate the centroids
        	for (int i = 0; i < numClusters; i++) {
        		if (countPosition[i] > 0) {
        			Instance newCentroid = new Instance(instanceLength);
        			for (int j = 0; j < instanceLength; j++) {
        				newCentroid.setValue(j,
        						(float) sumPosition[i][j] / countPosition[i]);
        			}
        			if (distFn.distance(newCentroid,
        					centroids[i]) > 0.0001) {
        				centroidsChanged = true;
        				centroids[i] = newCentroid;
        			}
        		}
        	}

        	// recalculate labels of the data
        	clusters = new int[data.numInstances()];
        	for (int i = 0; i < data.numInstances(); i++) {
        		int tmpCluster = 0;
        		double minDistance = distFn.distance(
        				centroids[0], data.instance(i));
        		for (int j = 0; j < centroids.length; j++) {
        			double dist = distFn.distance(
        					centroids[j], data.instance(i));
        			if (dist < minDistance) {
        				minDistance = dist;
        				tmpCluster = j;
        			}
        		}
        		clusters[i] = tmpCluster;
        	}
        }
	}
}
