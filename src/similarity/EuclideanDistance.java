package similarity;

import weka.core.Instance;
import weka.core.Instances;

/**
 * Implementation of the Euclidean Distance.
 *
 * @author Shalisa Pattarawuttiwong
 */
public class EuclideanDistance implements SimilarityFn {

  Instance x;
  Instance y;

  /**
   * Calculates the distance between two instances.
   *
   * @return        the euclidean distance between the two instances
   */
    public double calculateDistance() {
        if (x.numAttributes() != y.numAttributes()) {
            // throw exception
            System.out.println("Both instances should contain the same number of values");
        }
        double sum = 0.0;
        for (int i = 0; i <x.numAttributes(); i++) {
            if (!Double.isNaN(x.value(i)) && !Double.isNaN(y.value(i))) {
                sum += (y.value(i) - x.value(i)) * (y.value(i) - x.value(i));
            }
        }
        return Math.sqrt(sum);
  }

   public EuclideanDistance(Instance a, Instance b) {
        x = a;
        y = b;
   }

}
