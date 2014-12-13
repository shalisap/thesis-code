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
        KMeans kmeans = new KMeans(data, eucDist);
        kmeans.setNumClusters(1);
        kmeans.setNumIterations(100);
        kmeans.cluster();
        // test by number of clusters?
        assertEquals(1, getNumClusters(kmeans.getClusters()));
    }

    /**
     * Testing KMeans for two instances, one cluster using EuclideanDistance
     * Also need to have function to pick initial centroids
     */
    @Test
    public void twoInstancesOneClusterKMeansTest() throws Exception {
        readInInstances("./data/testTwo.arff");
        EuclideanDistance eucD = new EuclideanDistance();
        DistanceFunction eucDist = eucD;
        KMeans kmeans = new KMeans(data, eucDist);
        kmeans.setNumClusters(1);
        kmeans.setNumIterations(100);
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
        KMeans kmeans = new KMeans(data, eucDist);
        kmeans.setNumClusters(2);
        kmeans.setNumIterations(100);
        kmeans.cluster();
        assertEquals(2, getNumClusters(kmeans.getClusters()));
    }

   /**
     * Testing KMeans for three instances, one cluster using EuclideanDistance
     * three options with various initial centroids
     */
    @Test
    //@Ignore
    public void threeInstancesTwoClustersKMeansTest() throws Exception {
        readInInstances("./data/testThreeTwoCloser.arff");
        EuclideanDistance eucD = new EuclideanDistance();
        DistanceFunction eucDist = eucD;
        KMeans kmeans = new KMeans(data, eucDist);
        kmeans.setNumClusters(1);
        kmeans.setNumIterations(100);
        kmeans.cluster();
        assertEquals(1, getNumClusters(kmeans.getClusters()));
    }
	
	/**
	 * Test method for {@link clustering.KMeans#setNumClusters(int)}.
	 */
	@Test
	@Ignore
	public final void testSetNumClusters() {
	}

	/**
	 * Test method for {@link clustering.KMeans#setNumIterations(int)}.
	 */
	@Test
	@Ignore
	public final void testSetNumIterations() {
	}

	/**
	 * Test method for {@link clustering.KMeans#getClusters()}.
	 */
	@Test
	@Ignore
	public final void testGetClusters() {
	}
}
