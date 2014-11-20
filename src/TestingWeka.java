import java.io.*;
import java.util.*;

import weka.core.Instances;
import weka.core.Instance;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;
import similarity.EuclideanDistance;
import similarity.ManhattanDistance;

public class TestingWeka
{
    public static void main(String[] args) throws Exception
    {
        String infile = "./data/weather.numeric.arff";
        BufferedReader reader = new BufferedReader(new FileReader(infile));
        PrintWriter outfile = new PrintWriter(new FileWriter("./kMeans_output.txt"));
        Instances train = new Instances(reader);
        Instance a = train.instance(0);
        Instance b = train.instance(1);

        EuclideanDistance eucDist = new EuclideanDistance(a,b);
        System.out.println(eucDist.calculateDistance());

        ManhattanDistance manDist = new ManhattanDistance(a,b);
        System.out.println(manDist.calculateDistance());
    }
}
