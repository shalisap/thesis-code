package clustering;

import java.util.Random;
import java.util.Arrays;

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

    /**
     * Holds the labels for each instance in the data
     */
    private int[] output;

    /**
     * Set the number of clusters to find
     * @param k Number of clusters
     */
    public void setNumClusters(int k) {
         this.numClusters = k;
    }

    /**
     * Set the number of iterations to run
     * @param i Number of iterations
     */
    public void setNumIterations(int i) {
        this.iterations = i;
    }

    /**
     * Runs the kmeans clustering algorithm.
     * Implementation similar to the implementation of kmeans in the Java Machine Learning Library.
     */
    @Override
    public void cluster() {
        Random rand = new Random(); // random number generator

        if (this.data.numInstances() == 0) {
            System.out.println("The dataset should not be empty"); // should throw exception
        }
        if (numClusters == 0) {
            System.out.println("There should at least be one cluster"); // should throw exception
        }

        this.centroids = new Instances(this.data, numClusters);
        int instanceLength = this.data.instance(0).numAttributes();

        // Create instances that contain the min/max values for the attributes -- move to own function
        Instance min = new Instance(instanceLength);
        Instance max = new Instance(instanceLength);
        // for each instance
        for (int i = 0; i < this.data.numInstances(); i++) {
            Instance inst = this.data.instance(i);
            // for each attribute in the instance
            for (int j = 0; j < inst.numAttributes(); j++) {
                Attribute att = inst.attribute(j);
                double val = inst.value(j);
                if (max.isMissing(att) && min.isMissing(att)) {
                    max.setValue(j, val);
                    min.setValue(j, val);
                } else if (max.value(j) < val) {
                    max.setValue(j, val);
                } else if (min.value(j) > val)
                    min.setValue(j, val);
                }
        }

        //System.out.println("\nmin: " + min);
        //System.out.println("max: " + max);

        // Randomize centroids for first iteration
        for (int j = 0; j < numClusters; j++) {
            Instance randomInstance = this.data.instance(rand.nextInt(this.data.numInstances()));
            this.centroids.add(randomInstance);
        }

       // System.out.println("Beginning centroids: " + this.centroids.toString());

        int iterationCount = 0;
        boolean centroidsChanged = true;
        boolean randomCentroids = true;
        while (randomCentroids || (iterationCount < this.iterations && centroidsChanged)) {
            iterationCount++;
            int[] assignment = new int[this.data.numInstances()];

            if (iterationCount  == 1) {
                // assign each object to the group with the closest centroid
                for (int i = 0; i < this.data.numInstances(); i++) {
                    int tmpCluster = 0;
                    double minDistance = distFn.calculateDistance(this.centroids.instance(0), this.data.instance(i));
                    for (int j = 1; j < this.centroids.numInstances(); j++) {
                        double dist = distFn.calculateDistance(this.centroids.instance(j), this.data.instance(i));
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
            for (int i = 0; i < this.data.numInstances(); i++) {
                Instance in = this.data.instance(i);
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
                    Instance newCentroid = new Instance(instanceLength);
                    for (int j = 0; j < instanceLength; j++) {
                        newCentroid.setValue(j, (float) sumPosition[i][j] / countPosition[i]);
                    }
                    if (distFn.calculateDistance(newCentroid, this.centroids.instance(i)) > 0.0001) {
                        centroidsChanged = true;
                        // write a replace method in Instance.java?
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

            this.output = new int[this.data.numInstances()];
            for (int i = 0; i < this.data.numInstances(); i++) {
                int tmpCluster = 0;
                double minDistance = distFn.calculateDistance(centroids.instance(0), this.data.instance(i));
                for (int j = 0; j < this.centroids.numInstances(); j++) {
                    double dist = distFn.calculateDistance(this.centroids.instance(j), this.data.instance(i));
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

    /**
     * Returns the labels from kmeans clustering of the data
     */
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
