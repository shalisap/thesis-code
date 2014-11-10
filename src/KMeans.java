import java.io.*;
import java.util.*;

import weka.core.Instances;
import weka.clusterers.SimpleKMeans;
import weka.clusterers.ClusterEvaluation;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;

/**
 *  The KMeans class extends weka's SimpleKMeans and prints allows
 *  for removal of an attribute as well as printing of additional parameter.
 */
public class KMeans extends SimpleKMeans{

    /**
     * Removes the 'play' attribute from the weather.numeric.arff data set.
     * @param  train     The data
     * @return           The data without the 'play' attribute
     * @throws Exception [description]
     */
    private static Instances remove_attribute(Instances train) throws Exception {
            // remove play attribute
            String[] remove_op = new String[2];
            remove_op[0] = "-R"; // "range"
            remove_op[1] = "5"; // fifth attribute ('play')
            Remove remove = new Remove();
            remove.setOptions(remove_op);
            remove.setInputFormat(train);
            return Filter.useFilter(train, remove);
    }

    /**
     * Outputs a hashmap of <Integer : ArrayList<String>> which represents <Cluster : ArrayList of Instances in the specific cluster>
     * @param  newTrain  The data
     * @return           The hashmap of <Integer : ArrayList<String>>
     * @throws Exception [description]
     */
    private HashMap<Integer, ArrayList<String>> cluster_members(Instances newTrain) throws Exception {
             HashMap<Integer, ArrayList<String>> clusters = new HashMap<Integer, ArrayList<String>>();
            // keep hashmap of <cluster, instance>
            for (int i = 0; i < newTrain.numInstances(); i++) {
                int clusterNum = clusterInstance(newTrain.instance(i));

                if (clusters.containsKey(clusterNum)) {
                    ArrayList<String> group = clusters.get(clusterNum);
                    group.add(newTrain.instance(i).toString());
                    clusters.put(clusterNum, group);

                } else {
                    ArrayList<String> group = new ArrayList<String>();
                    group.add(newTrain.instance(i).toString());
                    clusters.put(clusterNum, group);
                }
            }
            return clusters;
    }

    /**
     * [kMeans description]
     * @throws Exception [description]
     */
    public static void kMeans() throws Exception {

        // load data
        String infile = "./data/weather.numeric.arff";
        BufferedReader reader = new BufferedReader(new FileReader(infile));
        PrintWriter outfile = new PrintWriter(new FileWriter("./kMeans_output.txt"));
        Instances train = new Instances(reader);
        Instances newTrain = remove_attribute(train);
        Instances test = newTrain;

        outfile.println("=== Run Information ===\n");
        outfile.println("Data file: " + infile);
        outfile.println("Instances: " + newTrain.numInstances());

        reader.close();

        // create model
        KMeans kmeans = new KMeans();
        kmeans.setNumClusters(2);
        kmeans.buildClusterer(newTrain);

        // print out clusters
        // print general info - evaluation
        ClusterEvaluation eval = new ClusterEvaluation();
        eval.setClusterer(kmeans);
        eval.evaluateClusterer(test);
        outfile.println(eval.clusterResultsToString());

        // print which cluster membership
        outfile.println("===Cluster Members===");
        HashMap<Integer, ArrayList<String>> clusters = kmeans.cluster_members(newTrain);

        for (Integer key: clusters.keySet()) {
            ArrayList<String> group = clusters.get(key);
            outfile.println("\n==Cluster " + key + ": " + group.size() + " members==\n");
            for (String s: group) {
                outfile.println(s);
            }
        }
        outfile.close();
    }

}
