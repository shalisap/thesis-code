package distance;

import java.lang.Math;
import weka.core.Instance;

/**
 * Implementation of the Edit Distance for two instances.
 * 
 * @author Shalisa Pattarawuttiwong
 */
public class EditDistance extends AbstractDistance implements DistanceFunction {

	/**
	 * The Edit Distance between two instances is the number of
	 * deletions, insertions, or substitutions required
	 * to transform one instance into another.
	 * 
	 * ed(m::ms, n::ns) =
	 * 	min(
	 * 		|m - n| + ed(ms, ns),
	 * 		m + ed(ms, n::ns),
	 * 		n + ed(m::ms, ns)
	 * 		)
	 *
	 *		where m, n are the attributes of each of the instances,
	 *		and where each attribute is positive (cell counts).
	 *
	 * @return the edit distance between the two instances
	 */
	@Override
	public double distance(Instance x, Instance y) {
		
	    int lengthX = x.numAttributes();
	    int lengthY = y.numAttributes();

	    // build a distance matrix 
	    double[][] disMatrix = new double[lengthX + 1][lengthY + 1];
	    
	    disMatrix[0][0] = 0;
	    
	    // fill in 0th row with the value of each attribute of x
	    // + previous attributes of x
	    for (int i = 1; i <= lengthX; i++) {
	    	disMatrix[i][0] = x.value(i - 1) + disMatrix[i-1][0];
	    }

	    // fill in 0th col with the value of each attribute of y
	    // + previous attributes of y
	    for (int j = 1; j <= lengthY; j++) {
	    	disMatrix[0][j] = y.value(j - 1) + disMatrix[0][j-1];
	    }

	    for (int i = 1; i <= lengthX; i++) {
	    	for (int j = 1; j <= lengthY; j++) {
	    		// -1 because 0th row/col filled 
	    		if (x.value(i - 1) == y.value(j - 1)) {  
	    			disMatrix[i][j] = disMatrix[i - 1][j - 1];
	    		} else {
	    			disMatrix[i][j] = Math.min((Math.abs(x.value(i - 1) - y.value(j - 1)) 
	        		  					+ disMatrix[i - 1][j - 1]),
	        		  					Math.min((x.value(i-1) + disMatrix[i - 1][j]),
	        		  							(y.value(j-1) + disMatrix[i][j - 1])));
		  	    }
	    	}
	    }
	    return disMatrix[lengthX][lengthY];
	}
	
	public EditDistance() {
	}

}
