package distance;

import weka.core.Instance;
import weka.core.Instances;

/**
 * Implementation of the Manhattan Distance.
 *
 * @author Shalisa Pattarawuttiwong
 */
public class ManhattanDistance extends AbstractDistance {

  Instance x;
  Instance y;

  /**
   * Calculates the distance between two instances.
   *
   * @return        the euclidean distance between the two instances
   */
    @Override
    public double calculateDistance() {
        if (x.numAttributes() != y.numAttributes()) {
            // throw exception
            System.out.println("Both instances should contain the same number of values");
        }
        double sum = 0.0;
        for (int i = 0; i <x.numAttributes(); i++) {
            sum += Math.abs(x.value(i) - y.value(i));
        }
        return sum;
  }

   public ManhattanDistance(Instance a, Instance b) {
        x = a;
        y = b;
   }

}
