import java.io.*;
import java.util.*;

import weka.core.Instances;
import weka.clusterers.SimpleKMeans;

public class KMeans {

    public static void main(String[] args) throws Exception {

        // load data
        try {
            String infile = "./data/weather.numeric.arff";
            BufferedReader reader = new BufferedReader(new FileReader(infile));
            PrintWriter outfile = new PrintWriter(new FileWriter("./kMeans_output.txt"));
            Instances data = new Instances(reader);

            outfile.println("=== Run Information ===\n");
            outfile.println("Data file: " + infile);
            outfile.println("Instances: " + data.numInstances());

            reader.close();

            // create model
            SimpleKMeans kmeans = new SimpleKMeans();
            kmeans.setNumClusters(2);
            kmeans.buildClusterer(data);

            // print out clusters
            // print general info
            outfile.print(kmeans);

            // print cluster centroids
            outfile.println("===Cluster Centroids===\n");
            Instances centroids = kmeans.getClusterCentroids();
            for (int i = 0; i < centroids.numInstances(); i++) {
                outfile.println("Centroid " + i + ": " + centroids.instance(i));
            }

            // print which cluster membership
            outfile.println("\n===Cluster Members===");

            HashMap<Integer, ArrayList<String>> clusters = new HashMap<Integer, ArrayList<String>>();
            // keep hashmap of <cluster, instance>
            for (int i = 0; i < data.numInstances(); i++) {
                int clusterNum = kmeans.clusterInstance(data.instance(i));

                if (clusters.containsKey(clusterNum)) {
                    ArrayList<String> group = clusters.get(clusterNum);
                    group.add(data.instance(i).toString());
                    clusters.put(clusterNum, group);

                } else {
                    ArrayList<String> group = new ArrayList<String>();
                    group.add(data.instance(i).toString());
                    clusters.put(clusterNum, group);
                }
            }

            // print membership
            for (Integer key: clusters.keySet()) {
                ArrayList<String> group = clusters.get(key);
                float percent = group.size() * 100f / data.numInstances();
                outfile.println("\n==Cluster " + key + ": " + group.size() + " members (" + percent + "%)==\n");
                for (String s: group) {
                    outfile.println(s);
                }
            }

            outfile.close();

        } catch (FileNotFoundException e) {
            System.out.println("File not found. Exception thrown:" + e);
        } catch (IOException e) {
            System.out.println("Exception thrown:" + e);
        }

    }

}
