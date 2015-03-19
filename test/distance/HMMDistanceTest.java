package distance;

import java.io.*;
import java.util.*;

import static org.junit.Assert.*;

import org.junit.Test;
//import org.junit.Ignore;


import be.ac.ulg.montefiore.run.jahmm.ObservationVector;
import weka.core.Instances;
import weka.core.Instance;

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
        System.out.println("Covariance Matrix");
        double[][] cov1 = hmmD.calcCovarianceMatrix(obs1, mean1);
        for (double[] c: cov1) {
        	System.out.println(Arrays.toString(c));
        }
        System.out.println("HMM");
        System.out.println(hmmD.initMultiHMM(data.instance(0), 2).toString());
        
        System.out.println("---------- ObservationSeq 2 ----------");
        List<ObservationVector> obs2 = hmmD.instanceToObservation(data.instance(1));
        System.out.println(obs2);
        System.out.println("Mean Vector");
        double[] mean2 = hmmD.calcVectorMean(obs2);
        System.out.println(Arrays.toString(mean2));
        System.out.println("Covariance Matrix");
        double[][] cov2 = hmmD.calcCovarianceMatrix(obs2, mean2);
        for (double[] c: cov2) {
        	System.out.println(Arrays.toString(c));
        }
        System.out.println("HMM");
        System.out.println(hmmD.initMultiHMM(data.instance(1), 2).toString());
        
        System.out.println("HMM Distance");
        double dist = hmmD.distance(data.instance(0), data.instance(1), 2);
        System.out.println(dist);
    }

}