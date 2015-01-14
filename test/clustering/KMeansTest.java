package clustering;

import distance.DistanceFunction;
import distance.EuclideanDistance;

import java.io.*;
import java.util.*;

import static org.junit.Assert.*;
import org.junit.Test;
import org.junit.Ignore;

import weka.core.Instances;
import weka.core.Instance;

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
     * Testing KMeans for one instance, one cluster using EuclideanDistance
     */
    @Test
    public void oneInstanceKMeansTest() throws Exception {
        readInInstances("./data/testSingle.arff");
        EuclideanDistance eucD = new EuclideanDistance();
        DistanceFunction eucDist = eucD;
        KMeans kmeans = new KMeans(this.data, eucDist);
        kmeans.setNumClusters(1);
        kmeans.setNumIterations(100);
        kmeans.setChooseInitCentroids(true);
        System.out.println("---------- One Instance, One Cluster (0 done) ----------");
        kmeans.cluster();
        // test by number of clusters?
        assertEquals(1, getNumClusters(kmeans.getClusters()));
    }

    /**
     * Testing KMeans for two instances, one cluster using EuclideanDistance
     * Picked the first instance as the centroid
     */
    @Test
    public void twoInstancesOneCluster1KMeansTest() throws Exception {
        readInInstances("./data/testTwo.arff");
        EuclideanDistance eucD = new EuclideanDistance();
        DistanceFunction eucDist = eucD;
        KMeans kmeans = new KMeans(this.data, eucDist);
        kmeans.setNumClusters(1);
        kmeans.setNumIterations(100);
        kmeans.setChooseInitCentroids(true);
        System.out.println("---------- Two Instances, One Cluster (0 done) ----------");
        kmeans.cluster();
        assertEquals(1, getNumClusters(kmeans.getClusters()));
    }
    
    /**
     * Testing KMeans for two instances, one cluster using EuclideanDistance
     * Picked the second instance as the centroid
     */
    @Test
    public void twoInstancesOneCluster2KMeansTest() throws Exception {
        readInInstances("./data/testTwo.arff");
        EuclideanDistance eucD = new EuclideanDistance();
        DistanceFunction eucDist = eucD;
        KMeans kmeans = new KMeans(this.data, eucDist);
        kmeans.setNumClusters(1);
        kmeans.setNumIterations(100);
        kmeans.setChooseInitCentroids(true);
        System.out.println("---------- Two Instances, One Cluster (1 done) ----------");
        kmeans.cluster();
        assertEquals(1, getNumClusters(kmeans.getClusters()));
    }


   /**
     * Testing KMeans for two instances, two clusters using EuclideanDistance
     */
    @Test
    public void twoInstancesTwoClustersKMeansTest() throws Exception {
        readInInstances("./data/testTwo.arff");
        EuclideanDistance eucD = new EuclideanDistance();
        DistanceFunction eucDist = eucD;
        KMeans kmeans = new KMeans(this.data, eucDist);
        kmeans.setNumClusters(2);
        kmeans.setNumIterations(100);
        kmeans.setChooseInitCentroids(true);
        System.out.println("---------- Two Instances, Two Clusters (0 1 done) ----------");
        kmeans.cluster();
        assertEquals(2, getNumClusters(kmeans.getClusters()));
    }

   /**
     * Testing KMeans for three instances, one cluster using EuclideanDistance
     * First 2 as initial centroids
     */
    @Test
    //@Ignore
    public void threeInstancesTwoClustersKMeansTest() throws Exception {
        readInInstances("./data/testThreeTwoCloser.arff");
        EuclideanDistance eucD = new EuclideanDistance();
        DistanceFunction eucDist = eucD;
        KMeans kmeans = new KMeans(this.data, eucDist);
        kmeans.setNumClusters(1);
        kmeans.setNumIterations(100);
        kmeans.setChooseInitCentroids(true);
        System.out.println("---------- Three Instances, Two Clusters (0 1 done) ----------");
        kmeans.cluster();
        assertEquals(1, getNumClusters(kmeans.getClusters()));
    }
	
    /**
     * Testing KMeans for three instances, one cluster using EuclideanDistance
     * Last 2 as initial centroids
     */
    @Test
    //@Ignore
    public void threeInstancesTwoClustersKMeansTest() throws Exception {
        readInInstances("./data/testThreeTwoCloser.arff");
        EuclideanDistance eucD = new EuclideanDistance();
        DistanceFunction eucDist = eucD;
        KMeans kmeans = new KMeans(this.data, eucDist);
        kmeans.setNumClusters(1);
        kmeans.setNumIterations(100);
        kmeans.setChooseInitCentroids(true);
        System.out.println("---------- Three Instances, Two Clusters (1 2 done) ----------");
        kmeans.cluster();
        assertEquals(1, getNumClusters(kmeans.getClusters()));
    }

    /**
     * Testing KMeans for three instances, one cluster using EuclideanDistance
     * First and Last as initial centroids
     */
    @Test
    //@Ignore
    public void threeInstancesTwoClustersKMeansTest() throws Exception {
        readInInstances("./data/testThreeTwoCloser.arff");
        EuclideanDistance eucD = new EuclideanDistance();
        DistanceFunction eucDist = eucD;
        KMeans kmeans = new KMeans(this.data, eucDist);
        kmeans.setNumClusters(1);
        kmeans.setNumIterations(100);
        kmeans.setChooseInitCentroids(true);
        System.out.println("---------- Three Instances, Two Clusters (0 2 done) ----------");
        kmeans.cluster();
        assertEquals(1, getNumClusters(kmeans.getClusters()));
    }