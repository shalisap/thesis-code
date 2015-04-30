package evaluation;

/**
 * @author Shalisa Pattarawuttiwong
 *
 * Code for various methods of evaluation of clusters.
 *
 */
public interface Evaluation {
	
	/**
	 * Given dataset D, and two labelings of D to clusters,
	 * returns the distance between the two.
	 *
	 * @param labels1 
	 * @param labels2
	 * @return distance between labels1 and labels2
	 */
	public double evaluate(int[] labels1, int[] labels2);

}
