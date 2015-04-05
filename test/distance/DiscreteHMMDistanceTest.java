package distance;

import java.io.*;

import org.junit.Test;
//import org.junit.Ignore;

import weka.core.Instances;

/**
 * Tests HMMDistance
 * 
 * @author Shalisa Pattarawuttiwong
 */
public class DiscreteHMMDistanceTest {
	
	
	private static Instances data;
	
    /**
     * Reads in instances from a .arff file
     * @param filename   name of the .arff file
     */
    public static void readInInstances(String filename)  throws Exception{
        BufferedReader reader = new BufferedReader(new FileReader(filename));
        data = new Instances(reader);
        
    }
    
    @Test
    //@Ignore
    public void HMMDistTest() throws Exception {
        readInInstances("./data/testMultiD.arff");
        DiscreteHMMDistance hmmD = new DiscreteHMMDistance(data.instance(0), data.instance(1), 3);
        System.out.println("Num HMM states: " + hmmD.states);
        System.out.print("HMM Distance: ");
        double dist = hmmD.distance(data.instance(0), data.instance(1));
        System.out.println(dist);
    }
    
    @Test
    public void DFHMMDistTest() throws Exception {
        readInInstances("./data/seriesdata50.arff");
        DiscreteHMMDistance hmmD = new DiscreteHMMDistance();
        hmmD.setNumStates(4);
        DistanceFunction distFn = hmmD;
        System.out.println("Num HMM states: " + hmmD.states);
        System.out.print("HMM Distance: ");
        double dist = distFn.distance(data.instance(0), data.instance(9));
        System.out.println(dist);
    }
}