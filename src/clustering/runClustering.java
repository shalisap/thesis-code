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
import distance.DistanceFunction;
import distance.EuclideanDistance;

/**
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
		    Map<Integer, int[]> clusters = new HashMap<Integer, int[]>();
			for (int k = min_k; k <= max_k; k++) {
		        readInInstances("./data/seriesdata.arff");
		        EuclideanDistance eucD = new EuclideanDistance();
		        DistanceFunction eucDist = eucD;
		        KMeans kmeans = new KMeans(data, eucDist);
		        kmeans.setNumClusters(k);
		        kmeans.setNumIterations(100);
		        kmeans.cluster();
		        clusters.put(k, kmeans.getClusters());
			}
			
			for (Map.Entry<Integer, int[]> entry : clusters.entrySet()) {
				System.out.println(entry.getKey());
				System.out.println(Arrays.toString(entry.getValue()));
			}
				
			Gson gson_out = new Gson();
			String json_out = gson_out.toJson(clusters);
			
			try {
				FileWriter file = new FileWriter("./data/clusters.json");
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
