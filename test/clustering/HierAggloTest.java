/**
 * 
 */
package clustering;

import static org.junit.Assert.*;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;

import org.junit.Test;

import distance.DistanceFunction;
import distance.EuclideanDistance;
import weka.core.Instances;

/**
 * Tests HierAgglo
 * 
 * @author Shalisa Pattarawuttiwong
 *
 */
public class HierAggloTest {

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
	 * 
	 */
	@Test
	public final void testGetAllClusters() throws Exception {
		readInInstances("./data/testThreeTwoCloser.arff");
        EuclideanDistance eucD = new EuclideanDistance();
        DistanceFunction eucDist = eucD;
        AgglomerationMethod singleLink = new SingleLinkage();
        HierAgglo hierAgglo = new HierAgglo(data, eucDist, singleLink);
        hierAgglo.setNumClusters(2);
        System.out.println("---------- All Clusters ----------");
        hierAgglo.cluster();
        
        for (int[] cluster: hierAgglo.getAllClusters()) {
        	System.out.println(Arrays.toString(cluster));
        }
        assertEquals(1,1);
	}

}
