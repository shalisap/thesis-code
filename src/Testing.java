import java.io.*;
import java.util.*;

import weka.core.Instances;
import weka.core.Instance;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;
import distance.EuclideanDistance;
import distance.ManhattanDistance;
//import clustering.KMeans;
import java.util.Random;

public class Testing
{
    public static void main(String[] args) throws Exception
    {
        String infile = "./data/weather.numeric.arff";
        BufferedReader reader = new BufferedReader(new FileReader(infile));
        PrintWriter outfile = new PrintWriter(new FileWriter("./kMeans_output.txt"));
        Instances train = new Instances(reader);
        Instance a = train.instance(0); // sunny,85,85,FALSE,no -> 0, 85, 85, 1, 1
        Instance b = train.instance(1); // sunny,80,90,TRUE,no -> 0, 80, 90, 0, 1
        System.out.println(a.toString());
        System.out.println(b.toString());

        // Testing Euclidean Distance
        double ans1 = Math.sqrt(Math.pow(85-80,2) + Math.pow(85-90,2) + Math.pow(1-0,2));
        EuclideanDistance eucDist = new EuclideanDistance(a,b);
        double test1 = eucDist.calculateDistance();
        if  (test1 != ans1) {
            System.out.println("Euclidean Distance: INCORRECT: got " + test1 + ", should be " + ans1);
        } else {
            System.out.println("Euclidean Distance: CORRECT");
        }

        // Testing Manhattan Distance
        double ans2 = (Math.abs(85-80) + Math.abs(85-90) + Math.abs(1-0));
        ManhattanDistance manDist = new ManhattanDistance(a,b);
        double test2 = manDist.calculateDistance();
        if  (test2 != ans2) {
            System.out.println("Manhattan Distance: INCORRECT: got " + test2 + ", should be " + ans2);
        } else {
            System.out.println("Manhattan Distance: CORRECT");
        }

        // Random
        Random rand = new Random();
        Instance randomInstance = train.instance(rand.nextInt(train.numInstances()));
        System.out.println(randomInstance.toString());

        // // Testing K Means
        // EuclideanDistance eucD = new EuclideanDistance();
        // KMeans kmeans = new KMeans(train, eucD);
        // kmeans.setNumClusters(2);
        // kmeans.setNumIterations(100);
        // kmeans.cluster();
        // System.out.println(kmeans.getClusters());

    }
}
