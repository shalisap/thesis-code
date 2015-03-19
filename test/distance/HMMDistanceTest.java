package distance;

import java.io.*;
import java.util.*;

import static org.junit.Assert.*;

import org.junit.Test;
//import org.junit.Ignore;

import be.ac.ulg.montefiore.run.jahmm.ObservationVector;
import weka.core.Instances;

/**
 * Tests HMMDistance
 * 
 * @author Shalisa Pattarawuttiwong
 */
public class HMMDistanceTest {
	
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
     * Testing HMMDist 
     */
    @Test
    //@Ignore
    public void twoInstancesHMMDistTest() throws Exception {
        System.out.println("---------- Two Instances ----------");
        readInInstances("./data/testMultiD.arff");
        HMMDistance hmmD = new HMMDistance();
        List<ObservationVector> obs1 = hmmD.instanceToObservation(data.instance(0));
        System.out.println("---------- ObservationSeq 1 ----------");
        System.out.println(obs1);
        System.out.println("Mean Vector");
        double[] mean1 = hmmD.calcVectorMean(obs1);
        System.out.println(Arrays.toString(mean1));
        double[] mean1Answer = new double[]{2.4, 2.6};
        for (int i = 0; i < mean1.length; i++) {
        	assertEquals(mean1Answer[i], mean1[i], 0.01);
        }
        
        System.out.println("Covariance Matrix");
        double[][] cov1 = hmmD.calcCovarianceMatrix(obs1, mean1);
        for (double[] c: cov1) {
        	System.out.println(Arrays.toString(c));
        }
        double[][] cov1Answer = new double[2][2];
        double[] cov1a = new double[]{5.44, 0.16};
        double[] cov1b = new double[]{0.16, 1.04};
        cov1Answer[0] = cov1a;
        cov1Answer[1] = cov1b;
        
        for (int i = 0; i < cov1.length; i++) {
            for (int j = 0; j < cov1.length; j++) {
        	assertEquals(cov1Answer[i][j], cov1[i][j], 0.01);
            }
        }
        
        System.out.println("HMM");
        System.out.println(hmmD.initMultiHMM(data.instance(0), 2).toString());
        
        System.out.println("---------- ObservationSeq 2 ----------");
        List<ObservationVector> obs2 = hmmD.instanceToObservation(data.instance(1));
        System.out.println(obs2);
        System.out.println("Mean Vector");
        double[] mean2 = hmmD.calcVectorMean(obs2);
        System.out.println(Arrays.toString(mean2));
        double[] mean2Answer = new double[]{2.4, 2.2};
        for (int i = 0; i < mean2.length; i++) {
        	assertEquals(mean2Answer[i], mean2[i], 0.01);
        }
        
        System.out.println("Covariance Matrix");
        double[][] cov2 = hmmD.calcCovarianceMatrix(obs2, mean2);
        for (double[] c: cov2) {
        	System.out.println(Arrays.toString(c));
        }
        double[][] cov2Answer = new double[2][2];
        double[] cov2a = new double[]{2.24, -0.48};
        double[] cov2b = new double[]{-0.48, 0.96};
        cov2Answer[0] = cov2a;
        cov2Answer[1] = cov2b;
        
        for (int i = 0; i < cov2.length; i++) {
            for (int j = 0; j < cov2.length; j++) {
        	assertEquals(cov2Answer[i][j], cov2[i][j], 0.01);
            }
        }
        
        
        System.out.println("HMM");
        System.out.println(hmmD.initMultiHMM(data.instance(1), 2).toString());
        
        System.out.println("HMM Distance");
        double dist = hmmD.distance(data.instance(0), data.instance(1), 2);
        System.out.println(dist);
    }

}