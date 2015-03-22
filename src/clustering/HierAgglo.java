package clustering;

import java.util.Arrays;

import weka.core.Instance;
import weka.core.Instances;
import distance.DistanceFunction;

/**
 * Implementation of Hierarchical Agglomerative Clustering.
 *
 * @author Shalisa Pattarawuttiwong
 */
public class HierAgglo implements ClusterAlg {

    /**
     * The similarity/distance function to be used
     */
    protected DistanceFunction distFn;

    /**
     * The data to be clustered
     */
    protected Instances data;
    
    /**
     * The agglomeration method to be used
     */
    protected AgglomerationMethod agglomerationMethod;

    /**
     * The number of clusters to generate
     */
    protected int numClusters = 2; // default value of k

    /**
     * The labels for each instance in the data, where 
     */
    
    private int[][] allClusters;
    
    /**
     * Set the number of clusters to generate
     * @param k Number of clusters
     */
    public void setNumClusters(int k) {
         this.numClusters = k;
    }

    /**
     * Runs the hierarchical agglomerative clustering algorithm.
     */
    @Override
    public void cluster() {
    	// distance matrix between data values.
    	final double[][] distMatrix = distFn.distMatrix(data);
    	final int numInstances = distMatrix.length;
    	//Initialize: first all in own clusters. Last all in same cluster
    	// Initialize trianglar array
    	allClusters = new int[numInstances][numInstances];
        //allClusters = new int[numInstances][];
        
        /**
        for (int i = 0; i < numInstances; i++) {
    		allClusters[i] = new int[numInstances - i];
        	for (int j = 0; j < numInstances - i; j++) {
        		System.out.println(j);
        		allClusters[i][j] = 0;	
        	}
        }
        
        // fill in largest array
        for (int i = 0; i < numInstances; i++) {
        	allClusters[0][i] = i;
        }
        
        System.out.println("J");
        for (int[] c: allClusters) {
        	System.out.println(Arrays.toString(c));
        }
        */
        
        for (int x = 0; x < numInstances; x++) {
        	allClusters[0][x] = x;
        	allClusters[numInstances - 1][x] = 0;
        }
        
        
    	final boolean[] indexUsed = new boolean[numInstances];
    	final int[] numPerCluster = new int[numInstances];
    	for (int i = 0; i < numInstances; i++) {
    		indexUsed[i] = true;
    		numPerCluster[i] = 1;
    	}
    	
    	// perform numInstances - 2 agglomerations? Don't do first or last.
    	for (int a = 1; a < numInstances - 1; a++) {
    		// determine the 2 most similar clusters
    		final int[] pair = findMostSimilarClusters(distMatrix, indexUsed);
    		final int i = Math.min(pair[0], pair[1]);
    		final int j = Math.max(pair[0], pair[1]);
    		final double d = distMatrix[i][j]; // get distance between the two 
    	
    		// cluster i is the new cluster
    		// agglomerates former clusters i and j, update distMatrix
    		for (int k = 0; k < numInstances; k++) {
    			if (k != i && k != j && indexUsed[k]) {
    				final double dist = agglomerationMethod.computeDist(distMatrix[i][k],
    						distMatrix[j][k], distMatrix[i][j], numPerCluster[i],
    						numPerCluster[j], numPerCluster[k]);
    				distMatrix[i][k] = dist;
    				distMatrix[k][i] = dist;
    			}
    		}
    		numPerCluster[i] = numPerCluster[i] + numPerCluster[j];
    	
    		// erase cluster j
    		indexUsed[j] = false;
    		for (int k = 0; k < numInstances; k++) {
    			distMatrix[j][k] = Double.POSITIVE_INFINITY;
    			distMatrix[k][j] = Double.POSITIVE_INFINITY; 
    		}
    	
    		// update clustering - first copy from previous row 
    		//allClusters[a] = Arrays.copyOf(allClusters[a - 1], 
    		//		allClusters[a - 1].length - 1);
    		
    		allClusters[a] = Arrays.copyOf(allClusters[a - 1], numInstances);
    		allClusters[a][j] = allClusters[a][i];
    		
    		// make sure there are only 0 to level of tree are used in that particular cluster
    		// if the new cluster is smaller than the max value at that cluster
    		if (allClusters[a][j] != numInstances - 1 - a) {
    			// existing clusters must merge
    			// everything larger must shift down 1.
    			for (int c = 0; c < numInstances; c++) {
    				if (allClusters[a][c] == allClusters[a - 1][j]) {
    					allClusters[a][c] = allClusters[a][i];
    				} else if (allClusters[a][c] > allClusters[a - 1][j]) {
    					allClusters[a][c] = allClusters[a][c] - 1;
    				}
    			}
    		} else {
    			// check again? 
    			for (int c = 0; c < numInstances; c++) {
    				if (allClusters[a][c] > numInstances - 1 - a) {
    					allClusters[a][c] = allClusters[a][c] - 1;
    				}
    			}
    		}
    	}
    }

    /**
     * Returns the pair of clusters that are the closest together.
     * @param distMatrix
     * @param indexUsed
     * @return
     */
    private static int[] findMostSimilarClusters(final double [][] distMatrix, 
    		final boolean[] indexUsed) {
    	final int[] mostSimilarPair = new int[2];
    	double smallestDist = Double.POSITIVE_INFINITY;
    	for (int cluster = 0; cluster < distMatrix.length; cluster++) {
    		if (indexUsed[cluster]) {
    			for (int neighbor = 0; neighbor < cluster; neighbor++) {
    				if (indexUsed[neighbor] && 
    						distMatrix[cluster][neighbor] < smallestDist) {  
    					smallestDist = distMatrix[cluster][neighbor];
    					mostSimilarPair[0] = cluster;
    					mostSimilarPair[1] = neighbor;
    				}
    			}
    		}
    	}
    	return mostSimilarPair;
    }
    
    /**
     * Returns the labels from hierarchical agglomerative clustering of the data
     * from the level specified.
     */
    @Override
    public int[] getClusters(){
		if (this.allClusters == null) {
			cluster();
		} 
        return this.allClusters[this.data.numInstances() - this.numClusters];
    }
    
    /**
     * Returns the entire tree with decreasing number of clusters
     * further down the array.
     */
    public int[][] getAllClusters(){
		if (this.allClusters == null) {
			cluster();
		} 
        return this.allClusters;
    }

   /**
    * Constructor for HierAgglo that takes data and
    * a similarity function.
    */
   public HierAgglo(Instances d, DistanceFunction s, AgglomerationMethod a) 
		   throws IllegalArgumentException {
        this.distFn = s;
        if (d.numInstances() <= 0) {
     		throw new IllegalArgumentException("The dataset"
     				+ " cannot be empty");
//        } else if (d.numAttributes() % 2 != 0) {
//     		throw new IllegalArgumentException("The dataset"
//     				+ " has an odd number of attributes. It must"
//     				+ " have pairs of (IN, OUT).");
        } else this.data = d;
        this.agglomerationMethod = a;
   }

}


