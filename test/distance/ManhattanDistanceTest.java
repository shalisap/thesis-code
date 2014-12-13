package distance;

import static org.junit.Assert.*;

import org.junit.Test;
import org.junit.Ignore;
//import org.junit.BeforeClass;

//import static org.mockito.Mockito.mock;
//import static org.mockito.Mockito.when;

import weka.core.Instance;

/**
 * Tests for ManhattanDistance
 * 
 * @author Shalisa Pattarawuttiwong
 */
public class ManhattanDistanceTest {

	private static Instance instance1;
	private static Instance instance2;
	
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
     * Testing the manhattan distance function when there are instances each with
     * a single positive attribute each.
     * The answer should be abs(15.0-10.0) = 5.00
     */
    @Test
    public final void testCalculateSingleAttributePositiveDistance() throws Exception{
        double [] attrs1 = {10.0};
        double [] attrs2 = {15.0};
        createInstances(attrs1, attrs2);
        ManhattanDistance manDist = new ManhattanDistance(instance1, instance2);
        double result = manDist.calculateDistance(instance1, instance2);
        assertEquals(5.00, result, 0.001);
    }

    /**
     * Testing the manhattan distance function when there are instances where
     * contains a single negative attribute.
     * The answer should be abs(15.0-(-10.0)) = 25.00
     */
    @Test
    public final void testCalculateSingleAttributeNegativeDistance() throws Exception{
        double [] attrs1 = {-10.0};
        double [] attrs2 = {15.0};
        createInstances(attrs1, attrs2);
        ManhattanDistance manDist = new ManhattanDistance(instance1, instance2);
        double result = manDist.calculateDistance(instance1, instance2);
        assertEquals(25.00, result, 0.001);
    }

    /**
     * Testing the manhattan distance function when there are instances each with
     * a single negative attribute.
     * The answer should be abs(-15.0-(-10.0)) = 5.00
     */
    @Test
    public final void testCalculateSingleAttribute2NegativeDistance() throws Exception{
        double [] attrs1 = {-10.0};
        double [] attrs2 = {-15.0};
        createInstances(attrs1, attrs2);
        ManhattanDistance manDist = new ManhattanDistance(instance1, instance2);
        double result = manDist.calculateDistance(instance1, instance2);
        assertEquals(5.00, result, 0.001);
    }

    /**
     * Testing the manhattan distance function when there are instances each with
     * multiple attributes.
     * The answer should be
     * abs((-10.0-15.0) + (8.5-(-4.2)) + (3.0-2.0)) = 38.7
     */
    @Test
    public final void testCalculateMultipleAttributeDistance() throws Exception{
        double [] attrs1 = {-10.0, 8.5, 3.0};
        double [] attrs2 = {15.0, -4.2, 2.0};
        createInstances(attrs1, attrs2);
        ManhattanDistance manDist = new ManhattanDistance(instance1, instance2);
        double result = manDist.calculateDistance(instance1, instance2);
        assertEquals(38.7, result, 0.001);
    }

    /**
     * Testing the euclidean distance function when there are instances with
     * uneven number of attributes.
     * An exception should be thrown. --- string printed?
     */
    @Test
    @Ignore
    public final void testCalculateUnevenInstancesDistance() throws Exception{
        double [] attrs1 = {-10.0, 3.00};
        double [] attrs2 = {15.0};
        createInstances(attrs1, attrs2);
        ManhattanDistance manDist = new ManhattanDistance(instance1, instance2);
        double result = manDist.calculateDistance(instance1, instance2);
        assertEquals("Both instances do not "
					+ "contain the same number of attributes", result);
    }


}
