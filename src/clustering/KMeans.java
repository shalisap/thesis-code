package clustering;

import java.util.Random;
import java.util.HashMap;
import java.util.Arrays;

import weka.classifiers.rules.DecisionTableHashKey; // class providing keys to the hash table
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Attribute;
import distance.DistanceFunction;

/**
 * Implementation of KMeans.
 *
 * @author Shalisa Pattarawuttiwong
 */
public class KMeans implements ClusterAlg {

    /**
     * Holds the similarity/distance function to be used
     */
    protected DistanceFunction distFn;

    /**
     * Holds the data to be processed
     */
    Instances data;

    int numClusters = 2; // default value of k; number of clusters to generate

    private int iterations = -1;

    /**
     * Holds the cluster centroids
     */
    private Instances centroids;

    private int[] output;

    public void setNumClusters(int k) {
         this.numClusters = k;
    }

    public void setNumIterations(int i) {
        this.iterations = i;
    }

    @Override
    public void cluster() {
        Random rand = new Random(); // random number generator

        if (data.numInstances() == 0) {
            System.out.println("The dataset should not be empty"); // should throw exception
        }
        if (numClusters == 0) {
            System.out.println("There should at least be one cluster"); // should throw exception
        }

        //this.centroids = new Instances(numClusters);
        this.centroids = new Instances(data, numClusters);
        int instanceLength = data.instance(0).numAttributes();

        // Create instances that contain the min/max values for the attributes -- move to own function
        Instance min = new Instance(instanceLength);
        Instance max = new Instance(instanceLength);
        // for each instance
        for (int i = 0; i < data.numInstances(); i++) {
            Instance inst = data.instance(i);
            // for each attribute in the instance
            for (int j = 0; j < inst.numAttributes(); j++) {
                Attribute att = inst.attribute(j);
                double val = inst.value(j);
                if (max.isMissing(att) && min.isMissing(att)) {
                    //max.put(j, att);
                    //min.put(j, att);
                    max.setValue(j, val);
                    min.setValue(j, val);
                } else if (max.value(j) < val) {
                    //max.put(j, att);
                    //max.deleteAttributeAt(j);
                    max.setValue(j, val);
                } else if (min.value(j) > val)
                    //min.put(j, att);
                    //min.deleteAttributeAt(j);
                    min.setValue(j, val);
                }
        }

        //System.out.println("\nmin: " + min);
        //System.out.println("max: " + max);

        // Randomize centroids for first iteration
        for (int j = 0; j < numClusters; j++) {
            Instance randomInstance = data.instance(rand.nextInt(data.numInstances()));
            this.centroids.add(randomInstance);
        }

        System.out.println("Beginning centroids: " + this.centroids.toString());

        int iterationCount = 0;
        boolean centroidsChanged = true;
        boolean randomCentroids = true;
        while (randomCentroids || (iterationCount < this.iterations && centroidsChanged)) {
            iterationCount++;
            int[] assignment = new int[data.numInstances()];

            if (iterationCount  == 1) {
                // assign each object to the group with the closest centroid
                for (int i = 0; i < data.numInstances(); i++) {
                    int tmpCluster = 0;
                    double minDistance = distFn.calculateDistance(this.centroids.instance(0), data.instance(i));
                    for (int j = 1; j < this.centroids.numInstances(); j++) {
                        double dist = distFn.calculateDistance(this.centroids.instance(j), data.instance(i));
                        if (distFn.compare(dist, minDistance)) {
                            minDistance = dist;
                            tmpCluster = j;
                        }
                    }
                    // assignment is an array of instances's cluster assignments
                    assignment[i] = tmpCluster;
                }
            }
            else {
                assignment = this.output;
            }

            //System.out.println("\nAssignment: " + Arrays.toString(assignment));

            // When all objects assigned, recalculate positions of K centroids, start over.
            // The new centroid is the weighted center of the current cluster
            double[][] sumPosition = new double[this.numClusters][instanceLength];
            // array of
            int[] countPosition = new int[this.numClusters];
            for (int i = 0; i < data.numInstances(); i++) {
                Instance in = data.instance(i);
                for (int j = 0; j < instanceLength; j++) {
                    // sumPosition is a double [[], []] of clusters of instances
                    sumPosition[assignment[i]][j] += in.value(j);
                }
                countPosition[assignment[i]]++;
            }

            // System.out.print("Sum Position: ");
            // for (int i = 0; i < sumPosition.length; i++) {
            //     System.out.print(Arrays.toString(sumPosition[i]));
            // }
            // System.out.print("\nCount Position: " + Arrays.toString(countPosition));

            centroidsChanged = false;
            randomCentroids = false;

            for (int i = 0; i < this.numClusters; i++) {
                if (countPosition[i] > 0) {
                    // double[] tmp = new double[instanceLength];
                    // for (int j = 0; j < instanceLength; j++) {
                    //     tmp[j] = (float) sumPosition[i][j] / countPosition[i];
                    // }
                    // Instance newCentroid = new Instance(tmp);
                    Instance newCentroid = new Instance(instanceLength);
                    for (int j = 0; j < instanceLength; j++) {
                        newCentroid.setValue(j, (float) sumPosition[i][j] / countPosition[i]);
                    }
                    if (distFn.calculateDistance(newCentroid, this.centroids.instance(i)) > 0.0001) {
                        centroidsChanged = true;
                        // write a replace method in Instance.java?
                        // centroids[i] = newCentroid;
                        this.centroids.delete(i);
                        this.centroids.add(newCentroid);
                    }
                } else {
                    Instance randomInstance = new Instance(instanceLength);
                    for (int j = 0; j < instanceLength; j++) {
                        double dist = Math.abs(max.value(j) - min.value(j));
                        randomInstance.setValue(j, (float) (min.value(j) + rand.nextDouble() * dist));
                    }
                    randomCentroids = true;
                    // replace
                    this.centroids.delete(i);
                    this.centroids.add(randomInstance);
                }
            }

              //System.out.println("\nRecalculated centroids: " + this.centroids.toString());

            this.output = new int[data.numInstances()];
            // for (int i = 0; i < centroids.numInstances(); i++) {
            //     output[i] = new int[instanceLength];
            // }
            for (int i = 0; i < data.numInstances(); i++) {
                int tmpCluster = 0;
                double minDistance = distFn.calculateDistance(centroids.instance(0), data.instance(i));
                for (int j = 0; j < this.centroids.numInstances(); j++) {
                    double dist = distFn.calculateDistance(this.centroids.instance(j), data.instance(i));
                    if (distFn.compare(dist, minDistance)) {
                        minDistance = dist;
                        tmpCluster = j;
                    }
                }
                this.output[i] = tmpCluster;
            }

            //System.out.println("\nAssignment 2: " + Arrays.toString(this.output));
        }
    }

    @Override
    public int[] getClusters(){
        return this.output;
    }

   /**
    * Constructor for KMeans that takes data and
    * a similarity function.
    */
   public KMeans(Instances d, DistanceFunction s) {
        this.distFn = s;
        this.data = d;
   }
}
