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
     * Determines the actual values in each cluster. 
     * Assumes that the maximum number of clusters is 4.
     */
    public ArrayList<ArrayList<String>> determineClusters(int[] clusters) {
    	ArrayList<ArrayList<String>> clusterValues = new ArrayList<ArrayList<String>>();
    	ArrayList<String> cluster0 = new ArrayList<String>();
    	ArrayList<String> cluster1 = new ArrayList<String>();
    	ArrayList<String> cluster2 = new ArrayList<String>();
    	ArrayList<String> cluster3 = new ArrayList<String>();
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
    		case 3:
    			cluster3.add(data.instance(numInst).toString());
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
    	Collections.sort(cluster3);
    	if (cluster0.size() > 0) clusterValues.add(cluster0);
    	if (cluster1.size() > 0) clusterValues.add(cluster1);
    	if (cluster2.size() > 0) clusterValues.add(cluster2);
    	if (cluster3.size() > 0) clusterValues.add(cluster3);
    	
    	Collections.sort(clusterValues, new Comparator<ArrayList<String>>() {
    		public int compare(ArrayList<String> a, ArrayList<String> b) {
    			return a.get(0).compareTo(b.get(0));
    		}
    	});
    	return clusterValues;
    }
    
	/**
	 * Testing HierAgglo for all clusters with 4 instances - single linkage
	 */
	@Test
	public final void testClusterSingleLink() throws Exception {
		readInInstances("./data/testFourThreeCloser.arff");
        EuclideanDistance eucD = new EuclideanDistance();
        DistanceFunction eucDist = eucD;
        AgglomerationMethod singleLink = new SingleLinkage();
        HierAgglo hierAgglo = new HierAgglo(data, eucDist, singleLink);
        System.out.println("---------- Single Linkage ----------");
        hierAgglo.cluster();
        
        for (int[] cluster: hierAgglo.getAllClusters()) {
        	System.out.println("level " + Arrays.toString(cluster));
        }
        
        // build expected results 
        // 4 clusters [[10], [11], [3], [8]]
        ArrayList<ArrayList<String>> expResult1 = new ArrayList<ArrayList<String>>();
        ArrayList<String> cluster0a = new ArrayList<String>();
        ArrayList<String> cluster1a = new ArrayList<String>();
        ArrayList<String> cluster2a = new ArrayList<String>();
        ArrayList<String> cluster3a = new ArrayList<String>();
        cluster0a.add(data.instance(0).toString());
        cluster1a.add(data.instance(1).toString());
        cluster2a.add(data.instance(2).toString());
        cluster3a.add(data.instance(3).toString());
        expResult1.add(cluster0a);
        expResult1.add(cluster1a);
        expResult1.add(cluster2a);
        expResult1.add(cluster3a);
    	Collections.sort(expResult1, new Comparator<ArrayList<String>>() {
    		public int compare(ArrayList<String> a, ArrayList<String> b) {
    			return a.get(0).compareTo(b.get(0));
    		}
    	});
    	
    	// 3 clusters [[10, 11], [3], [8]]
        ArrayList<ArrayList<String>> expResult2 = new ArrayList<ArrayList<String>>();
        ArrayList<String> cluster0b = new ArrayList<String>();
        ArrayList<String> cluster1b = new ArrayList<String>();
        ArrayList<String> cluster2b = new ArrayList<String>();
        cluster0b.add(data.instance(0).toString());
        cluster1b.add(data.instance(1).toString());
        cluster2b.add(data.instance(2).toString());
        cluster2b.add(data.instance(3).toString());
        Collections.sort(cluster2b);
        expResult2.add(cluster0b);
        expResult2.add(cluster1b);
        expResult2.add(cluster2b);
    	Collections.sort(expResult2, new Comparator<ArrayList<String>>() {
    		public int compare(ArrayList<String> a, ArrayList<String> b) {
    			return a.get(0).compareTo(b.get(0));
    		}
    	});
    	
    	// 2 clusters [[10, 11, 8], [3]]
        ArrayList<ArrayList<String>> expResult3 = new ArrayList<ArrayList<String>>();
        ArrayList<String> cluster0c = new ArrayList<String>();
        ArrayList<String> cluster1c = new ArrayList<String>();
        cluster0c.add(data.instance(0).toString());
        cluster1c.add(data.instance(1).toString());
        cluster1c.add(data.instance(2).toString());
        cluster1c.add(data.instance(3).toString());
        Collections.sort(cluster1c);
        expResult3.add(cluster0c);
        expResult3.add(cluster1c);
    	Collections.sort(expResult3, new Comparator<ArrayList<String>>() {
    		public int compare(ArrayList<String> a, ArrayList<String> b) {
    			return a.get(0).compareTo(b.get(0));
    		}
    	});
    	
    	// 1 cluster [[10, 11, 3, 8]]
        ArrayList<ArrayList<String>> expResult4 = new ArrayList<ArrayList<String>>();
        ArrayList<String> cluster0d = new ArrayList<String>();
        cluster0d.add(data.instance(0).toString());
        cluster0d.add(data.instance(1).toString());
        cluster0d.add(data.instance(2).toString());
        cluster0d.add(data.instance(3).toString());
        Collections.sort(cluster0d);
        expResult4.add(cluster0d);
    	Collections.sort(expResult4, new Comparator<ArrayList<String>>() {
    		public int compare(ArrayList<String> a, ArrayList<String> b) {
    			return a.get(0).compareTo(b.get(0));
    		}
    	});
        
    	hierAgglo.setNumClusters(4);
        assertEquals(expResult1, determineClusters(hierAgglo.getClusters()));
    	hierAgglo.setNumClusters(3);
        assertEquals(expResult2, determineClusters(hierAgglo.getClusters()));
    	hierAgglo.setNumClusters(2);
        assertEquals(expResult3, determineClusters(hierAgglo.getClusters()));
    	hierAgglo.setNumClusters(1);
        assertEquals(expResult4, determineClusters(hierAgglo.getClusters()));
	}

	/**
	 * Testing HierAgglo for all clusters with 4 instances - complete linkage
	 */
	@Test
	public final void testClusterCompleteLink() throws Exception {
		readInInstances("./data/testFourThreeCloser.arff");
        EuclideanDistance eucD = new EuclideanDistance();
        DistanceFunction eucDist = eucD;
        AgglomerationMethod completeLink = new CompleteLinkage();
        HierAgglo hierAgglo = new HierAgglo(data, eucDist, completeLink);
        System.out.println("---------- Complete Linkage ----------");
        hierAgglo.cluster();
        
        for (int[] cluster: hierAgglo.getAllClusters()) {
        	System.out.println("level " + Arrays.toString(cluster));
        }
        
        // build expected results 
        // 4 clusters [[10], [11], [3], [8]]
        ArrayList<ArrayList<String>> expResult1 = new ArrayList<ArrayList<String>>();
        ArrayList<String> cluster0a = new ArrayList<String>();
        ArrayList<String> cluster1a = new ArrayList<String>();
        ArrayList<String> cluster2a = new ArrayList<String>();
        ArrayList<String> cluster3a = new ArrayList<String>();
        cluster0a.add(data.instance(0).toString());
        cluster1a.add(data.instance(1).toString());
        cluster2a.add(data.instance(2).toString());
        cluster3a.add(data.instance(3).toString());
        expResult1.add(cluster0a);
        expResult1.add(cluster1a);
        expResult1.add(cluster2a);
        expResult1.add(cluster3a);
    	Collections.sort(expResult1, new Comparator<ArrayList<String>>() {
    		public int compare(ArrayList<String> a, ArrayList<String> b) {
    			return a.get(0).compareTo(b.get(0));
    		}
    	});
    	
    	// 3 clusters [[10, 11], [3], [8]]
        ArrayList<ArrayList<String>> expResult2 = new ArrayList<ArrayList<String>>();
        ArrayList<String> cluster0b = new ArrayList<String>();
        ArrayList<String> cluster1b = new ArrayList<String>();
        ArrayList<String> cluster2b = new ArrayList<String>();
        cluster0b.add(data.instance(0).toString());
        cluster1b.add(data.instance(1).toString());
        cluster2b.add(data.instance(2).toString());
        cluster2b.add(data.instance(3).toString());
        Collections.sort(cluster2b);
        expResult2.add(cluster0b);
        expResult2.add(cluster1b);
        expResult2.add(cluster2b);
    	Collections.sort(expResult2, new Comparator<ArrayList<String>>() {
    		public int compare(ArrayList<String> a, ArrayList<String> b) {
    			return a.get(0).compareTo(b.get(0));
    		}
    	});
    	
    	// 2 clusters [[10, 11, 8], [3]]
        ArrayList<ArrayList<String>> expResult3 = new ArrayList<ArrayList<String>>();
        ArrayList<String> cluster0c = new ArrayList<String>();
        ArrayList<String> cluster1c = new ArrayList<String>();
        cluster0c.add(data.instance(0).toString());
        cluster1c.add(data.instance(1).toString());
        cluster1c.add(data.instance(2).toString());
        cluster1c.add(data.instance(3).toString());
        Collections.sort(cluster1c);
        expResult3.add(cluster0c);
        expResult3.add(cluster1c);
    	Collections.sort(expResult3, new Comparator<ArrayList<String>>() {
    		public int compare(ArrayList<String> a, ArrayList<String> b) {
    			return a.get(0).compareTo(b.get(0));
    		}
    	});
    	
    	// 1 cluster [[10, 11, 3, 8]]
        ArrayList<ArrayList<String>> expResult4 = new ArrayList<ArrayList<String>>();
        ArrayList<String> cluster0d = new ArrayList<String>();
        cluster0d.add(data.instance(0).toString());
        cluster0d.add(data.instance(1).toString());
        cluster0d.add(data.instance(2).toString());
        cluster0d.add(data.instance(3).toString());
        Collections.sort(cluster0d);
        expResult4.add(cluster0d);
    	Collections.sort(expResult4, new Comparator<ArrayList<String>>() {
    		public int compare(ArrayList<String> a, ArrayList<String> b) {
    			return a.get(0).compareTo(b.get(0));
    		}
    	});
        
    	hierAgglo.setNumClusters(4);
        assertEquals(expResult1, determineClusters(hierAgglo.getClusters()));
    	hierAgglo.setNumClusters(3);
        assertEquals(expResult2, determineClusters(hierAgglo.getClusters()));
    	hierAgglo.setNumClusters(2);
        assertEquals(expResult3, determineClusters(hierAgglo.getClusters()));
    	hierAgglo.setNumClusters(1);
        assertEquals(expResult4, determineClusters(hierAgglo.getClusters()));
	}
	
	/**
	 * Testing HierAgglo for all clusters with 4 instances - group average method (UPGMA)
	 */
	@Test
	public final void testClusterGroupAverage() throws Exception {
		readInInstances("./data/testFourThreeCloser.arff");
        EuclideanDistance eucD = new EuclideanDistance();
        DistanceFunction eucDist = eucD;
        AgglomerationMethod groupAvg = new AverageLinkage();
        HierAgglo hierAgglo = new HierAgglo(data, eucDist, groupAvg);
        System.out.println("---------- Group Average (UPGMA) ----------");
        hierAgglo.cluster();
        
        for (int[] cluster: hierAgglo.getAllClusters()) {
        	System.out.println("level " + Arrays.toString(cluster));
        }
        
        // build expected results 
        // 4 clusters [[10], [11], [3], [8]]
        ArrayList<ArrayList<String>> expResult1 = new ArrayList<ArrayList<String>>();
        ArrayList<String> cluster0a = new ArrayList<String>();
        ArrayList<String> cluster1a = new ArrayList<String>();
        ArrayList<String> cluster2a = new ArrayList<String>();
        ArrayList<String> cluster3a = new ArrayList<String>();
        cluster0a.add(data.instance(0).toString());
        cluster1a.add(data.instance(1).toString());
        cluster2a.add(data.instance(2).toString());
        cluster3a.add(data.instance(3).toString());
        expResult1.add(cluster0a);
        expResult1.add(cluster1a);
        expResult1.add(cluster2a);
        expResult1.add(cluster3a);
    	Collections.sort(expResult1, new Comparator<ArrayList<String>>() {
    		public int compare(ArrayList<String> a, ArrayList<String> b) {
    			return a.get(0).compareTo(b.get(0));
    		}
    	});
    	
    	// 3 clusters [[10, 11], [3], [8]]
        ArrayList<ArrayList<String>> expResult2 = new ArrayList<ArrayList<String>>();
        ArrayList<String> cluster0b = new ArrayList<String>();
        ArrayList<String> cluster1b = new ArrayList<String>();
        ArrayList<String> cluster2b = new ArrayList<String>();
        cluster0b.add(data.instance(0).toString());
        cluster1b.add(data.instance(1).toString());
        cluster2b.add(data.instance(2).toString());
        cluster2b.add(data.instance(3).toString());
        Collections.sort(cluster2b);
        expResult2.add(cluster0b);
        expResult2.add(cluster1b);
        expResult2.add(cluster2b);
    	Collections.sort(expResult2, new Comparator<ArrayList<String>>() {
    		public int compare(ArrayList<String> a, ArrayList<String> b) {
    			return a.get(0).compareTo(b.get(0));
    		}
    	});
    	
    	// 2 clusters [[10, 11, 8], [3]]
        ArrayList<ArrayList<String>> expResult3 = new ArrayList<ArrayList<String>>();
        ArrayList<String> cluster0c = new ArrayList<String>();
        ArrayList<String> cluster1c = new ArrayList<String>();
        cluster0c.add(data.instance(0).toString());
        cluster1c.add(data.instance(1).toString());
        cluster1c.add(data.instance(2).toString());
        cluster1c.add(data.instance(3).toString());
        Collections.sort(cluster1c);
        expResult3.add(cluster0c);
        expResult3.add(cluster1c);
    	Collections.sort(expResult3, new Comparator<ArrayList<String>>() {
    		public int compare(ArrayList<String> a, ArrayList<String> b) {
    			return a.get(0).compareTo(b.get(0));
    		}
    	});
    	
    	// 1 cluster [[10, 11, 3, 8]]
        ArrayList<ArrayList<String>> expResult4 = new ArrayList<ArrayList<String>>();
        ArrayList<String> cluster0d = new ArrayList<String>();
        cluster0d.add(data.instance(0).toString());
        cluster0d.add(data.instance(1).toString());
        cluster0d.add(data.instance(2).toString());
        cluster0d.add(data.instance(3).toString());
        Collections.sort(cluster0d);
        expResult4.add(cluster0d);
    	Collections.sort(expResult4, new Comparator<ArrayList<String>>() {
    		public int compare(ArrayList<String> a, ArrayList<String> b) {
    			return a.get(0).compareTo(b.get(0));
    		}
    	});
        
    	hierAgglo.setNumClusters(4);
        assertEquals(expResult1, determineClusters(hierAgglo.getClusters()));
    	hierAgglo.setNumClusters(3);
        assertEquals(expResult2, determineClusters(hierAgglo.getClusters()));
    	hierAgglo.setNumClusters(2);
        assertEquals(expResult3, determineClusters(hierAgglo.getClusters()));
    	hierAgglo.setNumClusters(1);
        assertEquals(expResult4, determineClusters(hierAgglo.getClusters()));
	}
}
