package distance;

import java.lang.Math;
import java.util.Arrays;

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
		
	    int lengthX = x.numAttributes()/2;
	    int lengthY = y.numAttributes()/2;
		
		// convert to tuples of (in, out)
		double[] xAttrs = x.toDoubleArray();
		double[] yAttrs = y.toDoubleArray();
		
		double[][] xTup = new double[lengthX][2];
		double[][] yTup = new double[lengthY][2];
		int numX = 0;
		int numY = 0;
		for (int xs = 0; xs < lengthX; xs++) {
			double[] tup = new double[]{xAttrs[numX], xAttrs[numX + 1]};
			xTup[xs] = tup;
			numX = numX + 2;
		}
		for (int ys = 0; ys < lengthY; ys++) {
			double[] tup = new double[]{yAttrs[numY], yAttrs[numY + 1]};
			yTup[ys] = tup;
			numY = numY + 2;
		}

	    // build a distance matrix 
	    double[][] disMatrix = new double[lengthX + 1][lengthY + 1];
	    
	    disMatrix[0][0] = 0;
	    
	    // fill in 0th row with the value of each attribute of x
	    // + previous attributes of x
	    for (int i = 1; i <= lengthX; i++) {
	    	//disMatrix[i][0] = x.value(i - 1) + disMatrix[i-1][0];
	    	disMatrix[i][0] = xTup[i-1][0] + xTup[i-1][1] + disMatrix[i-1][0];
	    }

	    // fill in 0th col with the value of each attribute of y
	    // + previous attributes of y
	    for (int j = 1; j <= lengthY; j++) {
	    	//disMatrix[0][j] = y.value(j - 1) + disMatrix[0][j-1];
	    	disMatrix[0][j] = yTup[j-1][0] + yTup[j-1][1] + disMatrix[0][j-1];
	    }

	    for (int i = 1; i <= lengthX; i++) {
	    	for (int j = 1; j <= lengthY; j++) {
	    		// -1 because 0th row/col filled 
	    		if (Arrays.equals(xTup[i - 1], yTup[j - 1])) {  
	    			disMatrix[i][j] = disMatrix[i - 1][j - 1];
	    		} else {
	    			disMatrix[i][j] = Math.min((Math.abs(xTup[i-1][0] - yTup[j-1][0]) + 
	        		  					Math.abs(xTup[i-1][1] - yTup[j-1][1])
	    								+ disMatrix[i-1][j-1]),
	        		  					Math.min((xTup[i-1][0] + xTup[i-1][1] + disMatrix[i-1][j]),
	        		  							(yTup[j-1][0] + yTup[j-1][1] + disMatrix[i][j-1])));
		  	    }
	    	}
	    }
	    return disMatrix[lengthX][lengthY];
	}
	
	public EditDistance() {
	}

}
