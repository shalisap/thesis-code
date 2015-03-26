package distance;

import java.lang.Math;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

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
	 * ed((m_in, m_out)::ms, (n_in, n_out)::ns) =
	 * 	min(
	 * 		(|m_in - n_in| + |m_out - n_out|) + ed(ms, ns),
	 * 		(m_in + m_out) + ed(ms, n::ns),
	 * 		(n_in + m_out) + ed(m::ms, ns)
	 * 		)
	 *
	 *		where m, n are the attributes of each of the instances,
	 *		and where each attribute is positive (cell counts).
	 *
	 * @return the edit distance between the two instances
	 */
	@Override
	public double distance(Instance x, Instance y) {
		
		if (x.numAttributes() % 2 != 0 || y.numAttributes() % 2 != 0) {
			throw new IllegalArgumentException(
					"Number of attributes are not even");
		}
		
		// convert to tuples of (in, out)
		double[] xAttrs = x.toDoubleArray();
		double[] yAttrs = y.toDoubleArray();
		
		List<double[]> xTup = new ArrayList<double[]>();
		List<double[]> yTup = new ArrayList<double[]>();
		
		for (int numX = 0; numX < xAttrs.length; numX += 2) {
			// Ignore all destroy states
			if (xAttrs[numX] >= 0 && xAttrs[numX + 1] >= 0) {
				double[] tup = new double[]{xAttrs[numX], xAttrs[numX + 1]};
				xTup.add(tup);
			}
		}
		for (int numY = 0; numY < yAttrs.length; numY += 2) {
			if (xAttrs[numY] >= 0 && xAttrs[numY + 1] >= 0) {
				double[] tup = new double[]{yAttrs[numY], yAttrs[numY + 1]};
				yTup.add(tup);
			}
		}
		int lengthX = xTup.size();
		int lengthY = yTup.size();
		
	    // build a distance matrix 
	    double[][] disMatrix = new double[lengthX + 1][lengthY + 1];
	    
	    disMatrix[0][0] = 0;
	    
	    // fill in 0th row with the value of each attribute of x
	    // + previous attributes of x
	    for (int i = 1; i <= lengthX; i++) {
	    	//disMatrix[i][0] = x.value(i - 1) + disMatrix[i-1][0];
	    	disMatrix[i][0] = xTup.get(i-1)[0] + xTup.get(i-1)[1] 
	    			+ disMatrix[i-1][0];
	    }

	    // fill in 0th col with the value of each attribute of y
	    // + previous attributes of y
	    for (int j = 1; j <= lengthY; j++) {
	    	disMatrix[0][j] = yTup.get(j-1)[0] + yTup.get(j-1)[1] 
	    			+ disMatrix[0][j-1];
	    }

	    for (int i = 1; i <= lengthX; i++) {
	    	for (int j = 1; j <= lengthY; j++) {
	    		// -1 because 0th row/col filled 
				if (Arrays.equals(xTup.get(i - 1), yTup.get(j - 1))) {  
	    			disMatrix[i][j] = disMatrix[i - 1][j - 1];
	    		} else {
	    			double min = Math.min((
	    					Math.abs(xTup.get(i-1)[0] - yTup.get(j-1)[0]) 
	        		  		+ Math.abs(xTup.get(i-1)[1] - yTup.get(j-1)[1])
	    					+ disMatrix[i-1][j-1]),
	        		  		Math.min((xTup.get(i-1)[0] + xTup.get(i-1)[1] 
	        		  				+ disMatrix[i-1][j]),
	        		  				(yTup.get(j-1)[0] + yTup.get(j-1)[1] 
	        		  						+ disMatrix[i][j-1])));
	    			disMatrix[i][j] = min;
	    			
		  	    }
	    	}
	    }
	    return disMatrix[lengthX][lengthY];
	}
	
	public EditDistance() {
	}

}
