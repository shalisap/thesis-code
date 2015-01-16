package clustering;

import java.util.Random;
import java.util.Arrays;

import weka.core.Instance;
import weka.core.Instances;
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
    
    AgglomerationMethod agglomerationMethod;

    int numClusters = 2; // default value of k; number of clusters to generate

    /**
     * Holds the labels for each instance in the data
     */
    
    private int[][] allClusters;
    
    /**
     * Set the number of clusters to find
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
    	final double[][] distMatrix = distFn.calculateDistMatrix(data);
    	final int numInstances = distMatrix.length;
    	//Initialize: first all in own clusters. Last all in same cluster
        allClusters = new int[numInstances][numInstances];
        for (int x = 0; x < numInstances; x++) {
        	allClusters[0][x] = x;
        	allClusters[numInstances - 1][x] = 0;
        }
        
    	final boolean[] indexUsed = new boolean[numInstances];
    	final int[] clusterCardinalities = new int[numInstances];
    	for (int i = 0; i < numInstances; i++) {
    		indexUsed[i] = true;
    		clusterCardinalities[i] = 1;
    	}
    	
    	// perform numInstances - 2 agglomerations? Don't do first or last.
    	for (int a = 1; a < numInstances - 1; a++) {
    		// determine the 2 most similar clusters
    		final Pair pair = findMostSimilarClusters(distMatrix, indexUsed);
    		final int i = pair.getSmaller();
    		final int j = pair.getLarger();
    		final double d = distMatrix[i][j]; // get distance between the two 
    	
    		// cluster i is the new cluster
    		// agglomerates former clusters i and j, update distMatrix
    		for (int k = 0; k < numInstances; k++) {
    			if (k != i && k != j && indexUsed[k]) {
    				final double dist = agglomerationMethod.computeDist(distMatrix[i][k],
    						distMatrix[j][k], distMatrix[i][j], clusterCardinalities[i],
    						clusterCardinalities[j], clusterCardinalities[k]);
    				distMatrix[i][k] = dist;
    				distMatrix[k][i] = dist;
    			}
    		}
    		clusterCardinalities[i] = clusterCardinalities[i] + clusterCardinalities[j];
    	
    		// erase cluster j
    		indexUsed[j] = false;
    		for (int k = 0; k < numInstances; k++) {
    			distMatrix[j][k] = Double.POSITIVE_INFINITY;
    			distMatrix[k][j] = Double.POSITIVE_INFINITY; 
    		}
    	
    		// update clustering - first copy from previous row 
    		allClusters[a] = Arrays.copyOf(allClusters[a - 1], numInstances);
    		allClusters[a][j] = allClusters[a][i];
    		
    		// make sure there are only 0 to level of tree are used in that particular cluster
    		// if the new cluster is smaller than the max value at that cluster
    		if (allClusters[a][j] != numInstances - 1 - a) {
    			// everything larger must shift down 1.
    			for (int c = 0; c < numInstances; c++) {
    				if (allClusters[a][c] > allClusters[a][j]) {
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
    private static Pair findMostSimilarClusters(final double [][] distMatrix, 
    		final boolean[] indexUsed) {
    	final Pair mostSimilarPair = new Pair();
    	double smallestDist = Double.POSITIVE_INFINITY;
    	for (int cluster = 0; cluster < distMatrix.length; cluster++) {
    		if (indexUsed[cluster]) {
    			for (int neighbor = 0; neighbor < distMatrix.length; neighbor++) {
    				if (indexUsed[neighbor] && 
    						distMatrix[cluster][neighbor] < smallestDist && 
    						cluster != neighbor) {
    					smallestDist = distMatrix[cluster][neighbor];
    					mostSimilarPair.set(cluster, neighbor);
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
        return this.allClusters[numClusters];
    }
    
    /**
     * Returns the entire tree
     */
    public int[][] getAllClusters(){
        return this.allClusters;
    }

   /**
    * Constructor for HierAgglo that takes data and
    * a similarity function.
    */
   public HierAgglo(Instances d, DistanceFunction s, AgglomerationMethod a) {
        this.distFn = s;
        this.data = d;
        this.agglomerationMethod = a;
   }
   
   private static final class Pair {

       private int cluster1;
       private int cluster2;


       public final void set(final int cluster1, final int cluster2) {
           this.cluster1 = cluster1;
           this.cluster2 = cluster2;
       }

       public final int getLarger() {
           return Math.max(cluster1, cluster2);
       }

       public final int getSmaller() {
           return Math.min(cluster1, cluster2);
       }

   }
}


