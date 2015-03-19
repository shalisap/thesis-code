package distance;

import static org.junit.Assert.*;

import java.io.BufferedReader;
import java.io.FileReader;

import org.junit.Test;
//import org.junit.Ignore;
//import org.junit.BeforeClass;

//import static org.mockito.Mockito.mock;
//import static org.mockito.Mockito.when;

import weka.core.Instance;
import weka.core.Instances;
/**
 * Tests for EditDistance. Assume only positive attributes 
 * (to reflect cell counts). Ignore tests that test 
 * negative attributes.
 * 
 * @author Shalisa Pattarawuttiwong
 */
public class EditDistanceTest {
	
	private static Instance instance1;
	private static Instance instance2;
	private static Instances data;

	/**
	 * Creates instances to allow testing of various inputs
	 * @param attrs1  a list of attributes of instance1
	 * @param attrs2  a list of attributes of instance2
	 */
	public static void createInstances(double[] attrs1, double[] attrs2) {
		instance1 = new Instance(attrs1.length);
		instance2 = new Instance(attrs2.length);
		
		for (int i = 0; i < attrs1.length; i++) {
			instance1.setValue(i, attrs1[i]);
		}
		for (int i = 0; i < attrs2.length; i++) {
			instance2.setValue(i, attrs2[i]);
		}
	}

    /**
     * Reads in instances from a .arff file
     * @param filename   name of the .arff file
     */
    public static void readInInstances(String filename)  throws Exception{
        BufferedReader reader = new BufferedReader(new FileReader(filename));
        data = new Instances(reader);
    }
	
    /**
     * Testing the edit distance function when there are instances each with
     * a single positive attribute each.
     * The answer should be 5
     */
    @Test
    public final void testCalculateSingleAttributePositiveDistance() throws Exception{
        double [] attrs1 = {10.0};
        double [] attrs2 = {15.0};
        createInstances(attrs1, attrs2);
        EditDistance editDist = new EditDistance();
        System.out.println("Single Attribute");
        double result = editDist.distance(instance1, instance2);
        assertEquals(5, result, 0.001);
    }
    
    /**
     * Testing the edit distance function when there are instances each with
     * multiple attributes.
     */
    @Test
    public final void testCalculateMultipleAttributeDistance() throws Exception{
        double [] attrs1 = {10.0, 8.5, 5.0};
        double [] attrs2 = {15.0, 4.2, 40.0};
        createInstances(attrs1, attrs2);
        EditDistance editDist = new EditDistance();
        double result = editDist.distance(instance1, instance2);
        assertEquals(44.3, result, 0.001);
    }
    
    /**
     * Testing the calculate matrix function.
     */
    @Test
    public final void testCalculateDistMatrix() throws Exception{
    	// 3, 8, 10
    	readInInstances("./data/testThreeTwoCloser.arff");
    	double[][] expResult = {{0, 5, 7},{5, 0, 2},{7, 2, 0}};
    	EditDistance eucDist = new EditDistance();
    	double[][] calc = eucDist.distMatrix(data);
    	assertArrayEquals(expResult, calc);
    }
}
