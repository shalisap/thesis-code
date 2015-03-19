package clustering;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

import org.json.simple.JSONObject;
import org.json.simple.parser.JSONParser;

import com.google.gson.Gson;

import weka.core.Instances;
import distance.*;

/**
 * Driver file for executing clustering code
 * as a part of the pipeline for simulation
 * creation.
 * 
 * @author Shalisa Pattarawuttiwong
 */
public class runClustering {

	/**
	 * Data to be clustered
	 */
	private static Instances data;
	
    /**
     * Reads in instances from a .arff file
     * @param filename   name of the .arff file
     */
    public static void readInInstances(String filename)  throws Exception{
        BufferedReader reader = new BufferedReader(new FileReader(filename));
        data = new Instances(reader);
    }
	
	public static void main(String[] args) throws Exception{
		JSONParser parser = new JSONParser();
		
		try {
			// figure out min_k and max_k from json file
			Object obj = parser.parse(new FileReader(
					"./data/testing.json"));
			JSONObject jsonObject = (JSONObject) obj;
			int min_k = Integer.parseInt(jsonObject.get("min_k").toString());
			int max_k = Integer.parseInt(jsonObject.get("max_k").toString());
		    String cluster_alg = jsonObject.get("cluster_alg").toString();
			String dist_measure = jsonObject.get("dist_measure").toString();
			String arffpath = jsonObject.get("arffpath").toString();
			String cluster_outpath = jsonObject.get("cluster_outpath").toString();

			
			Map<Integer, int[]> clusters = new HashMap<Integer, int[]>();
			//for (int k = min_k; k <= max_k; k++) {
			int k = min_k;
			while (k <= max_k) {
		        readInInstances(arffpath);
		        
		        DistanceFunction distFn;
		        if (dist_measure.equals("euclidean")) {
		        	EuclideanDistance eucDist = new EuclideanDistance();
		        	distFn = eucDist;
		        } else if (dist_measure.equals("manhattan")) {
		        	ManhattanDistance manDist = new ManhattanDistance();
		        	distFn = manDist;
		        } else if (dist_measure.equals("edit distance")) {
		        	EditDistance editDist = new EditDistance();
		        	distFn = editDist;
			    } else {
		        	throw new IllegalArgumentException("No valid distance function "
		        			+ "chosen in .json config file.");
		        }

		        if (cluster_alg.equals("kmeans")) {
		        	KMeans kmeans = new KMeans(data, distFn);
			        kmeans.setNumClusters(k);
			        kmeans.setNumIterations(100);
			        kmeans.cluster();
			        clusters.put(k, kmeans.getClusters());
			        k++;
			        
		        } else if (cluster_alg.equals("kmedoids")) {
		        	KMedoids kmedoids = new KMedoids(data, distFn);
			        kmedoids.setNumClusters(k);
			        kmedoids.setNumIterations(100);
			        kmedoids.cluster();
			        clusters.put(k, kmedoids.getClusters());
			        k++;
			        
		        } else if (cluster_alg.equals("hierarchical")) {
		        	// need to add all agglomeration method options
		            AgglomerationMethod singleLink = new SingleLinkage();
		            HierAgglo hierAgglo = new HierAgglo(data, distFn, singleLink);
		            hierAgglo.cluster();
		            
		            for (int i = k; i <= max_k; i++) {
		            	hierAgglo.setNumClusters(i);
				        clusters.put(i, hierAgglo.getClusters());
		            }			
		            k = max_k + 1;
		            
//	            	int j = data.numInstances() - 1;
//	                for (int[] cluster: hierAgglo.getAllClusters()) {
//	                	System.out.println("level " + j + Arrays.toString(cluster));
//	                	j--;
//	                }
		        } else {
		        	throw new IllegalArgumentException("No valid clustering algorithm "
		        			+ "chosen in .json config file.");
		        }
		        
			}
			
			// print out clusters generated
			for (Map.Entry<Integer, int[]> entry : clusters.entrySet()) {
				System.out.println(entry.getKey());
				System.out.println(Arrays.toString(entry.getValue()));
			}
				
			// write out clusters in json format
			Gson gson_out = new Gson();
			String json_out = gson_out.toJson(clusters);
			
			try {
				FileWriter file = new FileWriter(cluster_outpath);
				file.write(json_out);
				file.flush();
				file.close();
			} catch (IOException e) {
				e.printStackTrace();
			}
			 
		} catch (Exception e) {
			e.printStackTrace();
		}
	}

}
