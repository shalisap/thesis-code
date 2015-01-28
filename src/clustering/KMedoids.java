package clustering;

import java.util.*;

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
     * Holds the cluster medoids
     */
    private Instance[] medoids;
    
    /**
     * If true, allows the user to pick the initial medoids,
     * if false, randomize the initial medoids.
     */
    private boolean chooseInitMedoids = false;

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
     * Choose to randomize or manually pick centroids.
     * @param i True for user's pick, false for randomize.
     */
    public void setChooseInitMedoids(boolean i) {
    	this.chooseInitMedoids = i;
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
    		double d = distFn.calculateDistance(inst, data.instance(i));
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
		// randomize initial medoids
    	int index = 0;
    	while (medoids.length < numClusters) {
    		boolean addRandom = true;
			Instance randInst = data.instance(rand.nextInt(data.numInstances()));
			// randomize first pick of medoids.
			if (Arrays.asList(medoids).contains(randInst)) {
				addRandom = false;
			}
			if (addRandom == true) {
				medoids[index] = randInst;
				index++;
			}
		}
    }
    
    /**
     * Allows the user to choose the initial medoids.
     */
    public void pickInitMedoids(){
    	// print instances
    	boolean duplicate = true;
    	ArrayList<String> indices = new ArrayList<String>();
    	
    	System.out.println("Instances");
    	for (int i = 0; i < data.numInstances(); i++) {
    		System.out.println(i + ": " + data.instance(i));
    	}
    	
		System.out.println("Enter indices of the distinct medoids wanted (Ex: 0 1 done): ");
    	Scanner input = new Scanner(System.in);
    	
    	int i = 0;
    	String index = input.next();
    	while ((!index.equals("done")) && duplicate == true) {
    		if (indices.contains(index)) {
    			duplicate = false;
    		}
    		else {
    			indices.add(index);
	    		Instance chosenInstance = data.instance(Integer.parseInt(index));	
	    		medoids[i] = chosenInstance;
	    		index = input.next();
	    		i++;
    		}
    	}
		System.out.println();
    }
    
	/** 
	 * 
	 */
	@Override
	public void cluster() {
		// initialize clusters to have max values
		clusters = new int[data.numInstances()];
    	medoids = new Instance[numClusters];
        if (chooseInitMedoids == false) {
        	randomizeInitMedoids();
        } else {
        	pickInitMedoids();
        	while (medoids.length != numClusters) {
        		System.out.println("Medoids chosen not equal to "
        				+ "the number of clusters wanted or "
        				+ "duplicates of centroids chosen");
                medoids = new Instance[numClusters];
        		pickInitMedoids();
        	}
        }
		
		boolean changed = true;
		int count = 0;
		while (changed && count < iterations) {
			changed = false;
			count++;
			
			// assign instances to medoids
			for (int i = 0; i < data.numInstances(); i++) {
				double bestDist = distFn.calculateDistance(data.instance(i), medoids[0]);
				int bestIndex = 0;
				for (int j = 1; j < medoids.length; j++) {
					double dist = distFn.calculateDistance(data.instance(i), medoids[j]);
						if (distFn.compare(dist, bestDist)) {
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
					medoids[m] = data.instance(rand.nextInt(data.numInstances()));
					// should make sure not duplicate
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
