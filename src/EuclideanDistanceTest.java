import distance.EuclideanDistance;

import org.junit.Test;
import org.junit.Ignore;
import org.junit.BeforeClass;
import static org.junit.Assert.*;
//import org.junit.runner.RunWith;
//import org.junit.runners.Junit4;

import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

import weka.core.Instance;

/**
 * Tests for EuclideanDistance
 *
 * @author  Shalisa Pattarawuttiwong
 */
public class EuclideanDistanceTest {

    private static Instance mockedInstance1;
    private static Instance mockedInstance2;

    /**
     * Creates mocks of two instances to allow testing of various inputs
     * @param attrs1   a list of attributes of instance1
     * @param attrs2   a list of attributes of instance2
     */
    public static void setUp(double[] attrs1, double[] attrs2) {
        mockedInstance1 = mock(Instance.class);
        mockedInstance2 = mock(Instance.class);
        when(mockedInstance1.numAttributes()).thenReturn(attrs1.length);
        when(mockedInstance2.numAttributes()).thenReturn(attrs2.length);

        for (int i = 0; i < attrs1.length; i++) {
            when(mockedInstance1.value(i)).thenReturn(attrs1[i]);
        }
        for (int i = 0; i < attrs2.length; i++) {
            when(mockedInstance2.value(i)).thenReturn(attrs2[i]);
        }
    }

    /**
     * Testing the euclidean distance function when there are instances each with
     * a single positive attribute each.
     * The answer should be sqrt((15.0-10.0)^2) = 5.00
     */
    @Test
    public void testCalculateSingleAttributePositiveDistance() throws Exception{
        double [] attrs1 = {10.0};
        double [] attrs2 = {15.0};
        setUp(attrs1, attrs2);
        EuclideanDistance eucDist = new EuclideanDistance(mockedInstance1, mockedInstance2);
        double result = eucDist.calculateDistance(mockedInstance1, mockedInstance2);
        assertEquals(5.00, result, 0.001);
    }

    /**
     * Testing the euclidean distance function when there are instances where
     * contains a single negative attribute.
     * The answer should be sqrt((15.0-(-10.0))^2) = 25.00
     */
    @Test
    public void testCalculateSingleAttributeNegativeDistance() throws Exception{
        double [] attrs1 = {-10.0};
        double [] attrs2 = {15.0};
        setUp(attrs1, attrs2);
        EuclideanDistance eucDist = new EuclideanDistance(mockedInstance1, mockedInstance2);
        double result = eucDist.calculateDistance(mockedInstance1, mockedInstance2);
        assertEquals(25.00, result, 0.001);
    }

    /**
     * Testing the euclidean distance function when there are instances each with
     * a single negative attribute.
     * The answer should be sqrt((-15.0-(-10.0))^2) = 5.00
     */
    @Test
    public void testCalculateSingleAttribute2NegativeDistance() throws Exception{
        double [] attrs1 = {-10.0};
        double [] attrs2 = {-15.0};
        setUp(attrs1, attrs2);
        EuclideanDistance eucDist = new EuclideanDistance(mockedInstance1, mockedInstance2);
        double result = eucDist.calculateDistance(mockedInstance1, mockedInstance2);
        assertEquals(5.00, result, 0.001);
    }

    /**
     * Testing the euclidean distance function when there are instances each with
     * multiple attributes.
     * The answer should be sqrt((-15.0-(-10.0))^2) = 5.00
     */
    @Test
    public void testCalculateMultipleAttributeDistance() throws Exception{
        double [] attrs1 = {-10.0, 8.5, 3.0};
        double [] attrs2 = {15.0, -4.2, 2.0};
        setUp(attrs1, attrs2);
        EuclideanDistance eucDist = new EuclideanDistance(mockedInstance1, mockedInstance2);
        double result = eucDist.calculateDistance(mockedInstance1, mockedInstance2);
        assertEquals(28.058688, result, 0.001);
    }

    /**
     * Testing the euclidean distance function when there are instances with
     * uneven number of attributes.
     * An exception should be thrown. --- string printed?
     */
    @Test
    @Ignore
    public void testCalculateUnevenInstancesDistance() throws Exception{
        double [] attrs1 = {-10.0, 3.00};
        double [] attrs2 = {15.0};
        setUp(attrs1, attrs2);
        EuclideanDistance eucDist = new EuclideanDistance(mockedInstance1, mockedInstance2);
        double result = eucDist.calculateDistance(mockedInstance1, mockedInstance2);
        assertEquals("Both instances should contain the same number of values", result);
    }

}
