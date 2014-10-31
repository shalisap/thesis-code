import java.io.*;

import weka.core.Instances;
import weka.clusterers.SimpleKMeans;

public class KMeans {

    public static void main(String[] args) throws Exception {

        // load data
        try {
            BufferedReader reader = new BufferedReader(new FileReader("./data/weather.numeric.arff"));
            Instances data = new Instances(reader);
            reader.close();

            // create model
            SimpleKMeans kmeans = new SimpleKMeans();
            kmeans.setNumClusters(2);
            kmeans.buildClusterer(data);

            // print out cluster centroids
            Instances centroids = kmeans.getClusterCentroids();
            for (int i = 0; i < data.numInstances(); i++) {
                System.out.print(data.instance(i));
                System.out.print(" is in cluster ");
                System.out.println(kmeans.clusterInstance(data.instance(i)) + 1);
            }

            System.out.println(kmeans);

        } catch (FileNotFoundException e) {
            System.out.println("File not found. Exception thrown:" + e);
        } catch (IOException e) {
            System.out.println("Exception thrown:" + e);
        }

    }

}
