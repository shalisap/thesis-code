package clustering;

import java.util.*;

import weka.core.Instance;
import weka.core.Instances;
import distance.DistanceFunction;

/**
 * Implementation of K-Medoids clustering.
 * 
 * @author Shalisa Pattarawuttiwong
 */
public class KMedoids implements ClusterAlg {
	
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
     * The cluster medoids, [m{0}, m{1}, ..., m{n-1}], where
     * each c{i}, 0 <= m < n, is an instance representing 
     * a medoid and n is the number of clusters.
     */
    private Instance[] medoids;
    
    /**
     * [i{0}, i{1}, ..., i{n-1}] where each i{j}, 0 <= j < n belongs 
     * to a partition, C{k}, 0 <= k <= numClusters, of the dataset. 
     */
    private int[] clusters;
    
    /**
     * Random number generator 
     */
    private Random rand;
    
    /**
     * Constructor for KMedoids that takes data and
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
     * Creates an instance that contains the average values 
     * for the attributes of the data.
     * 
     * @return Instance representing the average attribute values
     */
    private Instance averageInstance(Instances insts) {
    	Instance avgInst = new Instance(insts.numAttributes());
    	for (int i = 0; i < insts.numAttributes(); i++) {
    		double sum = 0;
    		for (int j = 0; j < insts.numInstances(); j++) {
    			sum += insts.instance(j).value(i);
    		}
    		avgInst.setValue(i, sum/insts.numInstances());
    	}
    	return avgInst;
    }
    
    /**
     * Returns an instance of the data that is
     * the closest to the instance given as a parameter
     * 
     * @param inst Instance to find the closest instance
     * @return instance closest to the given instance
     */
    public Instance closestInst(Instance inst) {
    	int closest = 0;
    	double min = Double.POSITIVE_INFINITY;
    	for (int i = 0; i < data.numInstances(); i++) {
    		double d = distFn.distance(inst, data.instance(i));
    		if (d < min && !inst.equals(data.instance(i))) {
    			closest = i;
    			min = d; // if d < min, make min = d.
    		}
    	}
    	return data.instance(closest);
    }
    
    /**
     * Randomizes the initial medoids chosen. 
     */
    public void randomizeInitMedoids(){
        medoids = new Instance[this.numClusters];
        
        // construct list xs where xs[i] = i
        List<Integer> random = new ArrayList<Integer>();
        for (int i = 0; i < data.numInstances(); i++) {
        	random.add(i);
        }
        
        // shuffle xs
        Collections.shuffle(random);
        
        for (int k = 0; k < numClusters; k++) {
        	medoids[k] = data.instance(random.get(k));
        }
    }
    
    /**
     * Allows the user to choose the initial medoids.
     * @param pickedMed Set of integers representing indices of the data
     */
    public void setInitMedoids(Set<Integer> pickedMed){
    	
        medoids = new Instance[this.numClusters];
        int index = 0;
    	for (int item: pickedMed) {
    		medoids[index] = data.instance(item);
    		//centroids.add(this.data.instance(item));
    		index++;
    	}
  
    	if (medoids.length != numClusters) {
    		throw new IllegalArgumentException("Centroids chosen not equal to "
    				+ "the number of clusters wanted or "
    				+ "duplicates of centroids chosen");
    	}
    
    }
    
    /**
     * k-means++ to choose initial medoids.
     */
    private void kMeansPlusPlusInit(){
    	medoids = new Instance[this.numClusters];
    	boolean[] taken = new boolean[data.numInstances()];
    	
    	// choose one center at random
    	int idx = rand.nextInt(data.numInstances());
    	Instance center = data.instance(idx);
    	this.medoids[0] = center;
    	// mark index as taken
    	taken[idx] = true;
    	
    	// keep track of min distance squared of elements of data
    	// to elements of medoids
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
    			this.medoids[chosen] = nextCenter;
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
	 * Returns the labels from KMedoids clustering of the data
	 */
	@Override
	public int[] getClusters() {
		if (this.clusters == null) {
			cluster();
		} 
		return this.clusters;
	}
    
	/** 
	 * Runs the k-medoids clustering algorithm on the data given.
	 */
	@Override
	public void cluster() {
		// initialize clusters to have max values
		clusters = new int[data.numInstances()];

        if (this.medoids == null) {
        	//randomizeInitMedoids();
        	kMeansPlusPlusInit();
        }
        
		boolean changed = true;
		int count = 0;
		while (changed && count < iterations) {
			changed = false;
			count++;
			
			// assign instances to medoids
			for (int i = 0; i < data.numInstances(); i++) {
				double bestDist = distFn.distance(data.instance(i), medoids[0]);
				int bestIndex = 0;
				for (int j = 1; j < medoids.length; j++) {
					double dist = distFn.distance(data.instance(i), medoids[j]);
						if (dist < bestDist) {
							bestDist = dist;
							bestIndex = j;
						}
				}
				clusters[i] = bestIndex;
			}
			
			// figure out actual clusters
			Instances[] actClusters = new Instances[numClusters];
			for (int i = 0; i < numClusters; i++) {
				Instances cluster = new Instances(data, data.numInstances());
				for (int j = 0; j < clusters.length; j++) {
					if (clusters[j] == i) {
						cluster.add(data.instance(j));
					}
				}
				actClusters[i] = cluster;
			}
			
			// recalculate medoids
			for (int m = 0; m < numClusters; m++) {
				if (actClusters[m].numInstances() == 0) { // new random, empty medoid
					// should make sure not duplicate
					int randInt = rand.nextInt(data.numInstances());
					while (!Arrays.asList(medoids).contains(data.instance(randInt))) {
						medoids[m] = data.instance(randInt);
					}
					//medoids.instance(i) = data.instance(rand.nextInt(data.numInstances()));
					changed = true;
				} else {
					Instance centroid = averageInstance(actClusters[m]);
					Instance oldMedoid = medoids[m];
					Instance closest = closestInst(centroid);
					medoids[m] = closest;
					if (!medoids[m].equals(oldMedoid))
						changed = true;
				}
			}
		}
	}

}
