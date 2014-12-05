package distance;

/**
 * Abstract super class for all distance measures. High values denote more
 * distant instances.
 *
 * @author  Shalisa Pattarawuttiwong
 */
public abstract class AbstractDistance implements DistanceFunction {

    @Override
    public boolean compare(double x, double y) {
        return x < y;
    }
}
