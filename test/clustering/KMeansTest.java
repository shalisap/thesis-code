package clustering;

import distance.DistanceFunction;
import distance.EuclideanDistance;

import java.io.*;
import java.util.*;

import static org.junit.Assert.*;
import org.junit.Test;
//import org.junit.Ignore;

import weka.core.Instances;
//import weka.core.Instance;

/**
 * Tests KMeans
 * 
 * @author Shalisa Pattarawuttiwong
 */
public class KMeansTest {
	
	private static Instances data;
	
    /**
     * Reads in instances from a .arff file
     * @param filename   name of the .arff file
     */
    public static void readInInstances(String filename)  throws Exception{
        BufferedReader reader = new BufferedReader(new FileReader(filename));
        data = new Instances(reader);
    }
    
    /**
     * Returns the number of different clusters from an assignment of clusters
     * @param  clusters assignment of instances into different clusters
     * @return          integer of different clusters
     */
    public int getNumClusters(int[] clusters) {
        ArrayList<Integer> num = new ArrayList<Integer>();
        for (int i : clusters) {
            if (!num.contains(i)) {
                num.add(i);
            }
        }
        return num.size();
    }
    
    /**
     * Determines the actual values in each cluster. 
     * Assumes that the maximum number of clusters is 3.
     */
    public ArrayList<ArrayList<String>> determineClusters(int[] clusters) {
    	ArrayList<ArrayList<String>> clusterValues = new ArrayList<ArrayList<String>>();
    	ArrayList<String> cluster0 = new ArrayList<String>();
    	ArrayList<String> cluster1 = new ArrayList<String>();
    	ArrayList<String> cluster2 = new ArrayList<String>();
    	int numInst = 0;
    	for (int i : clusters) {
    		switch (i) {
    		case 0:
    			cluster0.add(data.instance(numInst).toString());
    			break;
    		case 1:
    			cluster1.add(data.instance(numInst).toString());
    			break;
    		case 2:
    			cluster2.add(data.instance(numInst).toString());
    			break;
    		default:
    			System.out.println("Instance assigned to cluster that does not exist?");
    			break;
    		}
    		numInst++;
    	}
    	Collections.sort(cluster0);
    	Collections.sort(cluster1);
    	Collections.sort(cluster2);
    	if (cluster0.size() > 0) clusterValues.add(cluster0);
    	if (cluster1.size() > 0) clusterValues.add(cluster1);
    	if (cluster2.size() > 0) clusterValues.add(cluster2);
    	
    	Collections.sort(clusterValues, new Comparator<ArrayList<String>>() {
    		public int compare(ArrayList<String> a, ArrayList<String> b) {
    			return a.get(0).compareTo(b.get(0));
    		}
    	});
    	return clusterValues;
    }
    
    /**
     * Testing KMeans for one instance, one cluster using EuclideanDistance
     */
    @Test
    //@Ignore
    public void oneInstanceKMeansTest() throws Exception {
        readInInstances("./data/testSingle.arff");
        EuclideanDistance eucD = new EuclideanDistance();
        DistanceFunction eucDist = eucD;
        KMeans kmeans = new KMeans(data, eucDist);
        kmeans.setNumClusters(1);
        kmeans.setNumIterations(100);
        //kmeans.setInitCentroids([0]);
        Set<Integer> init = new HashSet<Integer>();
        init.add(0);
        kmeans.setInitCentroids(init);
        System.out.println("---------- One Instance, One Cluster "
        		+ "(0 done) ----------");
        kmeans.cluster();
        
        System.out.println(Arrays.toString(kmeans.getClusters()));
        System.out.println(determineClusters(kmeans.getClusters()));
        // test number of clusters
        assertEquals(1, getNumClusters(kmeans.getClusters()));
        
        // build expected result [[10]]
        ArrayList<ArrayList<String>> expResult = new ArrayList<ArrayList<String>>();
        ArrayList<String> cluster0 = new ArrayList<String>();
        cluster0.add(data.instance(0).toString());
        expResult.add(cluster0);
        
        assertEquals(expResult, determineClusters(kmeans.getClusters()));
    }

    /**
     * Testing KMeans for two instances, one cluster using EuclideanDistance
     * Picked the first instance as the centroid
     */
    @Test
    //@Ignore
    public void twoInstancesOneCluster1KMeansTest() throws Exception {
        readInInstances("./data/testTwo.arff");
        EuclideanDistance eucD = new EuclideanDistance();
        DistanceFunction eucDist = eucD;
        KMeans kmeans = new KMeans(data, eucDist);
        kmeans.setNumClusters(1);
        kmeans.setNumIterations(100);
        Set<Integer> init = new HashSet<Integer>();
        init.add(0);
        kmeans.setInitCentroids(init);
        System.out.println("---------- Two Instances, One Cluster "
        		+ "(0 done) ----------");
        kmeans.cluster();
        
        System.out.println(Arrays.toString(kmeans.getClusters()));
        System.out.println(determineClusters(kmeans.getClusters()));
        
        //test number of clusters
        assertEquals(1, getNumClusters(kmeans.getClusters()));
        
        // build expected result [[3, 8]]
        ArrayList<ArrayList<String>> expResult = new ArrayList<ArrayList<String>>();
        ArrayList<String> cluster0 = new ArrayList<String>();
        cluster0.add(data.instance(0).toString());
        cluster0.add(data.instance(1).toString());
        expResult.add(cluster0);
        
        assertEquals(expResult, determineClusters(kmeans.getClusters()));
    }
    
    /**
     * Testing KMeans for two instances, one cluster using EuclideanDistance
     * Picked the second instance as the centroid
     */
    @Test
    //@Ignore
    public void twoInstancesOneCluster2KMeansTest() throws Exception {
        readInInstances("./data/testTwo.arff");
        EuclideanDistance eucD = new EuclideanDistance();
        DistanceFunction eucDist = eucD;
        KMeans kmeans = new KMeans(data, eucDist);
        kmeans.setNumClusters(1);
        kmeans.setNumIterations(100);
        Set<Integer> init = new HashSet<Integer>();
        init.add(1);
        kmeans.setInitCentroids(init);
        System.out.println("---------- Two Instances, One Cluster "
        		+ "(1 done) ----------");
        kmeans.cluster();
        
        System.out.println(Arrays.toString(kmeans.getClusters()));
        System.out.println(determineClusters(kmeans.getClusters()));
        
        //test number of clusters
        assertEquals(1, getNumClusters(kmeans.getClusters()));
        
        // build expected result [[3, 8]]
        ArrayList<ArrayList<String>> expResult = new ArrayList<ArrayList<String>>();
        ArrayList<String> cluster0 = new ArrayList<String>();
        cluster0.add(data.instance(0).toString());
        cluster0.add(data.instance(1).toString());
        expResult.add(cluster0);
        
        assertEquals(expResult, determineClusters(kmeans.getClusters()));
    }


   /**
     * Testing KMeans for two instances, two clusters using EuclideanDistance
     */
    @Test
    //@Ignore
    public void twoInstancesTwoClustersKMeansTest() throws Exception {
        readInInstances("./data/testTwo.arff");
        EuclideanDistance eucD = new EuclideanDistance();
        DistanceFunction eucDist = eucD;
        KMeans kmeans = new KMeans(data, eucDist);
        kmeans.setNumClusters(2);
        kmeans.setNumIterations(100);
        Set<Integer> init = new HashSet<Integer>();
        init.add(0);
        init.add(1);
        kmeans.setInitCentroids(init);
        System.out.println("---------- Two Instances, Two Clusters "
        		+ "(0 1 done) ----------");
        kmeans.cluster();
        
        System.out.println(Arrays.toString(kmeans.getClusters()));
        System.out.println(determineClusters(kmeans.getClusters()));
        
        //test number of clusters
        assertEquals(2, getNumClusters(kmeans.getClusters()));
        
        // build expected result [[3], [8]]
        ArrayList<ArrayList<String>> expResult = new ArrayList<ArrayList<String>>();
        ArrayList<String> cluster0 = new ArrayList<String>();
        ArrayList<String> cluster1 = new ArrayList<String>();
        cluster0.add(data.instance(0).toString());
        cluster1.add(data.instance(1).toString());
        expResult.add(cluster0);
        expResult.add(cluster1);
        
        assertEquals(expResult, determineClusters(kmeans.getClusters()));
    }

    /**
     * Testing KMeans for three instances, one clusters using EuclideanDistance
     * First as initial centroid
     */
    @Test
    //@Ignore
    public void threeInstancesOneCluster1KMeansTest() throws Exception {
        readInInstances("./data/testThreeTwoCloser.arff");
        EuclideanDistance eucD = new EuclideanDistance();
        DistanceFunction eucDist = eucD;
        KMeans kmeans = new KMeans(data, eucDist);
        kmeans.setNumClusters(1);
        kmeans.setNumIterations(100);
        Set<Integer> init = new HashSet<Integer>();
        init.add(0);
        kmeans.setInitCentroids(init);
        System.out.println("---------- Three Instances, One Cluster"
        		+ " (0 done) ----------");
        kmeans.cluster();
        
        System.out.println(Arrays.toString(kmeans.getClusters()));
        System.out.println(determineClusters(kmeans.getClusters()));
        
        //test number of clusters
        assertEquals(1, getNumClusters(kmeans.getClusters()));
        
        // build expected result [[10, 3, 8]]
        ArrayList<ArrayList<String>> expResult = new ArrayList<ArrayList<String>>();
        ArrayList<String> cluster0 = new ArrayList<String>();
        cluster0.add(data.instance(0).toString());
        cluster0.add(data.instance(1).toString());
        cluster0.add(data.instance(2).toString());
        Collections.sort(cluster0);
        expResult.add(cluster0);
        
        assertEquals(expResult, determineClusters(kmeans.getClusters()));
    }
    
    /**
     * Testing KMeans for three instances, one clusters using EuclideanDistance
     * Second as initial centroid
     */
    @Test
    //@Ignore
    public void threeInstancesOneCluster2KMeansTest() throws Exception {
        readInInstances("./data/testThreeTwoCloser.arff");
        EuclideanDistance eucD = new EuclideanDistance();
        DistanceFunction eucDist = eucD;
        KMeans kmeans = new KMeans(data, eucDist);
        kmeans.setNumClusters(1);
        kmeans.setNumIterations(100);
        Set<Integer> init = new HashSet<Integer>();
        init.add(1);
        kmeans.setInitCentroids(init);
        System.out.println("---------- Three Instances, One Cluster"
        		+ " (1 done) ----------");
        kmeans.cluster();
        
        System.out.println(Arrays.toString(kmeans.getClusters()));
        System.out.println(determineClusters(kmeans.getClusters()));
        
        //test number of clusters
        assertEquals(1, getNumClusters(kmeans.getClusters()));
        
        // build expected result [[10, 3, 8]]
        ArrayList<ArrayList<String>> expResult = new ArrayList<ArrayList<String>>();
        ArrayList<String> cluster0 = new ArrayList<String>();
        cluster0.add(data.instance(0).toString());
        cluster0.add(data.instance(1).toString());
        cluster0.add(data.instance(2).toString());
        Collections.sort(cluster0);
        expResult.add(cluster0);
        
        assertEquals(expResult, determineClusters(kmeans.getClusters()));
    }
    
    /**
     * Testing KMeans for three instances, one clusters using EuclideanDistance
     * Last as initial centroid
     */
    @Test
    //@Ignore
    public void threeInstancesOneCluster3KMeansTest() throws Exception {
        readInInstances("./data/testThreeTwoCloser.arff");
        EuclideanDistance eucD = new EuclideanDistance();
        DistanceFunction eucDist = eucD;
        KMeans kmeans = new KMeans(data, eucDist);
        kmeans.setNumClusters(1);
        kmeans.setNumIterations(100);
        Set<Integer> init = new HashSet<Integer>();
        init.add(2);
        kmeans.setInitCentroids(init);
        System.out.println("---------- Three Instances, One Cluster "
        		+ "(2 done) ----------");
        kmeans.cluster();
        
        System.out.println(Arrays.toString(kmeans.getClusters()));
        System.out.println(determineClusters(kmeans.getClusters()));
        
        //test number of clusters
        assertEquals(1, getNumClusters(kmeans.getClusters()));
        
        // build expected result [[10, 3, 8]]
        ArrayList<ArrayList<String>> expResult = new ArrayList<ArrayList<String>>();
        ArrayList<String> cluster0 = new ArrayList<String>();
        cluster0.add(data.instance(0).toString());
        cluster0.add(data.instance(1).toString());
        cluster0.add(data.instance(2).toString());
        Collections.sort(cluster0);
        expResult.add(cluster0);
        
        assertEquals(expResult, determineClusters(kmeans.getClusters()));
    }
    
   /**
     * Testing KMeans for three instances, two clusters using EuclideanDistance
     * First 2 as initial centroids
     */
    @Test
    //@Ignore
    public void threeInstancesTwoClusters1KMeansTest() throws Exception {
        readInInstances("./data/testThreeTwoCloser.arff");
        EuclideanDistance eucD = new EuclideanDistance();
        DistanceFunction eucDist = eucD;
        KMeans kmeans = new KMeans(data, eucDist);
        kmeans.setNumClusters(2);
        kmeans.setNumIterations(100);
        Set<Integer> init = new HashSet<Integer>();
        init.add(0);
        init.add(1);
        kmeans.setInitCentroids(init);
        System.out.println("---------- Three Instances, Two Clusters "
        		+ "(0 1 done) ----------");
        kmeans.cluster();
        
        System.out.println(Arrays.toString(kmeans.getClusters()));
        System.out.println(determineClusters(kmeans.getClusters()));
        
        //test number of clusters
        assertEquals(2, getNumClusters(kmeans.getClusters()));
        
        // build expected result [[10, 8], [3]]
        ArrayList<ArrayList<String>> expResult = new ArrayList<ArrayList<String>>();
        ArrayList<String> cluster0 = new ArrayList<String>();
        ArrayList<String> cluster1 = new ArrayList<String>();
        cluster0.add(data.instance(0).toString());
        cluster1.add(data.instance(1).toString());
        cluster1.add(data.instance(2).toString());
        Collections.sort(cluster0);
        Collections.sort(cluster1);
        expResult.add(cluster0);
        expResult.add(cluster1);
    	Collections.sort(expResult, new Comparator<ArrayList<String>>() {
    		public int compare(ArrayList<String> a, ArrayList<String> b) {
    			return a.get(0).compareTo(b.get(0));
    		}
    	});
        
        assertEquals(expResult, determineClusters(kmeans.getClusters()));
    }
	
    /**
     * Testing KMeans for three instances, two clusters using EuclideanDistance
     * Last 2 as initial centroids
     */
    @Test
    //@Ignore
    public void threeInstancesTwoClusters2KMeansTest() throws Exception {
        readInInstances("./data/testThreeTwoCloser.arff");
        EuclideanDistance eucD = new EuclideanDistance();
        DistanceFunction eucDist = eucD;
        KMeans kmeans = new KMeans(data, eucDist);
        kmeans.setNumClusters(2);
        kmeans.setNumIterations(100);
        Set<Integer> init = new HashSet<Integer>();
        init.add(1);
        init.add(2);
        kmeans.setInitCentroids(init);
        System.out.println("---------- Three Instances, Two Clusters "
        		+ "(1 2 done) ----------");
        kmeans.cluster();
        
        System.out.println(Arrays.toString(kmeans.getClusters()));
        System.out.println(determineClusters(kmeans.getClusters()));
        
        //test number of clusters
        assertEquals(2, getNumClusters(kmeans.getClusters()));
        
        // build expected result [[10, 8], [3]]
        ArrayList<ArrayList<String>> expResult = new ArrayList<ArrayList<String>>();
        ArrayList<String> cluster0 = new ArrayList<String>();
        ArrayList<String> cluster1 = new ArrayList<String>();
        cluster0.add(data.instance(0).toString());
        cluster1.add(data.instance(1).toString());
        cluster1.add(data.instance(2).toString());
        Collections.sort(cluster0);
        Collections.sort(cluster1);
        expResult.add(cluster0);
        expResult.add(cluster1);
    	Collections.sort(expResult, new Comparator<ArrayList<String>>() {
    		public int compare(ArrayList<String> a, ArrayList<String> b) {
    			return a.get(0).compareTo(b.get(0));
    		}
    	});
        
        assertEquals(expResult, determineClusters(kmeans.getClusters()));
    }

    /**
     * Testing KMeans for three instances, two clusters using EuclideanDistance
     * First and Last as initial centroids
     */
    @Test
    //@Ignore
    public void threeInstancesTwoClusters3KMeansTest() throws Exception {
        readInInstances("./data/testThreeTwoCloser.arff");
        EuclideanDistance eucD = new EuclideanDistance();
        DistanceFunction eucDist = eucD;
        KMeans kmeans = new KMeans(data, eucDist);
        kmeans.setNumClusters(2);
        kmeans.setNumIterations(100);
        Set<Integer> init = new HashSet<Integer>();
        init.add(0);
        init.add(2);
        kmeans.setInitCentroids(init);
        System.out.println("---------- Three Instances, Two Clusters"
        		+ " (0 2 done) ----------");
        kmeans.cluster();
        
        System.out.println(Arrays.toString(kmeans.getClusters()));
        System.out.println(determineClusters(kmeans.getClusters()));
        
        //test number of clusters
        assertEquals(2, getNumClusters(kmeans.getClusters()));
        
        // build expected result [[10, 8], [3]]
        ArrayList<ArrayList<String>> expResult = new ArrayList<ArrayList<String>>();
        ArrayList<String> cluster0 = new ArrayList<String>();
        ArrayList<String> cluster1 = new ArrayList<String>();
        cluster0.add(data.instance(0).toString());
        cluster1.add(data.instance(1).toString());
        cluster1.add(data.instance(2).toString());
        Collections.sort(cluster0);
        Collections.sort(cluster1);
        expResult.add(cluster0);
        expResult.add(cluster1);
    	Collections.sort(expResult, new Comparator<ArrayList<String>>() {
    		public int compare(ArrayList<String> a, ArrayList<String> b) {
    			return a.get(0).compareTo(b.get(0));
    		}
    	});
        
        assertEquals(expResult, determineClusters(kmeans.getClusters()));
    }
    
    /**
     * Testing KMeans for three instances, three clusters using EuclideanDistance
     */
    @Test
    //@Ignore
    public void threeInstancesThreeClustersKMeansTest() throws Exception {
        readInInstances("./data/testThreeTwoCloser.arff");
        EuclideanDistance eucD = new EuclideanDistance();
        DistanceFunction eucDist = eucD;
        KMeans kmeans = new KMeans(data, eucDist);
        kmeans.setNumClusters(3);
        kmeans.setNumIterations(100);
        Set<Integer> init = new HashSet<Integer>();
        init.add(0);
        init.add(1);
        init.add(2);
        kmeans.setInitCentroids(init);
        System.out.println("---------- Three Instances, three Clusters "
        		+ "(0 1 2 done) ----------");
        kmeans.cluster();
        
        System.out.println(Arrays.toString(kmeans.getClusters()));
        System.out.println(determineClusters(kmeans.getClusters()));
        
        //test number of clusters
        assertEquals(3, getNumClusters(kmeans.getClusters()));
        
        // build expected result [[10], [3], [8]]
        ArrayList<ArrayList<String>> expResult = new ArrayList<ArrayList<String>>();
        ArrayList<String> cluster0 = new ArrayList<String>();
        ArrayList<String> cluster1 = new ArrayList<String>();
        ArrayList<String> cluster2 = new ArrayList<String>();
        cluster0.add(data.instance(0).toString());
        cluster1.add(data.instance(1).toString());
        cluster2.add(data.instance(2).toString());
        Collections.sort(cluster0);
        Collections.sort(cluster1);
        Collections.sort(cluster2);
        expResult.add(cluster0);
        expResult.add(cluster1);
        expResult.add(cluster2);
    	Collections.sort(expResult, new Comparator<ArrayList<String>>() {
    		public int compare(ArrayList<String> a, ArrayList<String> b) {
    			return a.get(0).compareTo(b.get(0));
    		}
    	});
        
        assertEquals(expResult, determineClusters(kmeans.getClusters()));
    }
}