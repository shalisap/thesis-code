import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.TimeUnit;

import org.json.simple.JSONObject;
import org.json.simple.parser.JSONParser;

import weka.core.Instances;
import clustering.AgglomerationMethod;
import clustering.AverageLinkage;
import clustering.CentroidLinkage;
import clustering.CompleteLinkage;
import clustering.HierAgglo;
import clustering.KMeans;
import clustering.KMedoids;
import clustering.MedianLinkage;
import clustering.SingleLinkage;
import clustering.WardLinkage;
import clustering.WeightedAverageLinkage;

import com.google.gson.Gson;

import distance.DiscreteHMMDistance;
import distance.DistanceFunction;
import distance.EditDistance;
import distance.EuclideanDistance;
import distance.ManhattanDistance;
import evaluation.CollapsedPairs;
import evaluation.DistinguishingPairs;
import evaluation.DistinguishingPairsAdj;


public class FiveClusters {


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
		String gt_outpath = "shadow-500r-1800c_ground_truth.json";
		//String gt_outpath = "./data/shadow-50r-180c_ground_truth.json";
		String arffpath = "./data/shadow-500r-1800c/1/seriesdata.arff";
		//String arffpath = "./data/seriesdata.arff";
		String outpath = "./data/newClusters";
		
        readInInstances(arffpath);
		JSONParser subparser = new JSONParser();
		
		// grab ground truth from ground truth .json 
		int[] ground_truth = new int[data.numInstances()];
		try {
			Object subobj = subparser.parse(new FileReader(gt_outpath));
			JSONObject subjsonObject = (JSONObject) subobj;
			String gt_list = subjsonObject.get("ground truth").toString();
			
			Gson gson = new Gson();
			ground_truth = gson.fromJson(gt_list, int[].class);
		} catch (Exception e) {
			e.printStackTrace();
		}
		
		// Find clustering.json files in clusters directory
		String path = "./data"; 

		String filesString;
		File folder = new File(path);
		File[] listOfFiles = folder.listFiles(); 
		List<String> files = new ArrayList<String>();
		
		// get list of cluster .json files
		for (int i = 0; i < listOfFiles.length; i++) 
		{

			if (listOfFiles[i].isFile()) 
			{
				filesString = listOfFiles[i].getName();
				if (filesString.endsWith("clusters.json"))
				{
					files.add(path + "/" + filesString);
				}
			}
		}
		System.out.println(files);
		// for each file, grab cluster labels and put in list
		for (int i = 0; i < files.size(); i++) {
			int[][] clusters = new int[9][data.numInstances()];
			try {
				String[] split = files.get(i).split("_");
				System.out.println(Arrays.toString(split));
				
				String clust = "";
				String dist = "";
				String agglo = "";
				int distIdx = 1;
				if (split[0].contains("hier")) {
					clust = "hierarchical";
					distIdx = 2;
					if (split[1].contains("wavg")) {
						agglo = "weighted average";
					} else if (split[1].contains("cent")) {
						agglo = "centroid";
					} else if (split[1].contains("comp")) {
						agglo = "complete";
					} else if (split[1].contains("med")) {
						agglo = "median";
					} else if (split[1].contains("single")) {
						agglo = "single";
					} else if (split[1].contains("ward")) {
						agglo = "ward";
					} else if (split[1].contains("avg")) {
						agglo = "average";
					} else {
						System.out.println("NOOOO AGGLO METHOD");
					}
				} else if (split[0].contains("kmeans")) {
					clust = "kmeans";
				} else if (split[0].contains("kmedoids")) {
					clust = "kmedoids";
				}
				
				if (split[distIdx].contains("euc")) {
					dist = "euclidean";
				} else if (split[distIdx].contains("man")) {
					dist = "manhattan";
				} else if (split[distIdx].contains("edit")) {
					dist = "edit";
				}
				String newFile = outpath + "/" + clust + "-" + dist + ".log";
				System.out.println(newFile);
				PrintWriter writer = new PrintWriter(newFile);
				
				writer.println("Clustering Algorithm: " + clust);
				writer.println("Distance Measure: " + dist);
				
				if (!agglo.equals("")) {
					writer.println("Agglomeration Method: " + agglo);
				}
				
				Object subobj = subparser.parse(new FileReader(files.get(i)));
				JSONObject subjsonObject = (JSONObject) subobj;
				
				// grab all 9 clusters and put in list
				String c_list_2 = subjsonObject.get("2").toString();
				String c_list_3 = subjsonObject.get("3").toString();
				String c_list_4 = subjsonObject.get("4").toString();
				String c_list_5 = subjsonObject.get("5").toString();
				String c_list_6 = subjsonObject.get("6").toString();
				String c_list_7 = subjsonObject.get("7").toString();
				String c_list_8 = subjsonObject.get("8").toString();
				String c_list_9 = subjsonObject.get("9").toString();

				Gson gson = new Gson();
				 clusters[0] = gson.fromJson(c_list_2, int[].class);
				 clusters[1] = gson.fromJson(c_list_3, int[].class);
				 clusters[2] = gson.fromJson(c_list_4, int[].class);
				 clusters[3] = gson.fromJson(c_list_5, int[].class);
				 clusters[4] = gson.fromJson(c_list_6, int[].class);
				 clusters[5] = gson.fromJson(c_list_7, int[].class);
				 clusters[6] = gson.fromJson(c_list_8, int[].class);
				 clusters[7] = gson.fromJson(c_list_9, int[].class);

				 
				 int num = 2;
				 for (int c = 0; c < clusters.length; c++) {
					 writer.println("Cluster " + num);
			         DistinguishingPairs dp = new DistinguishingPairs();
			         DistinguishingPairsAdj dpAdj = new DistinguishingPairsAdj();
			         CollapsedPairs cp = new CollapsedPairs();
			        	
			         double dp_eval = dp.evaluate(clusters[c], ground_truth);
			         double dpAdj_eval = dpAdj.evaluate(clusters[c], ground_truth);
			         double cp_eval = cp.evaluate(clusters[c], ground_truth);
			        	
			         writer.println("Distinguishing Pairs (Rand Index): " + dp_eval);
			         writer.println("Distinguishing Pairs Adjusted (Adjusted Rand Index): " + dpAdj_eval);
			         writer.println("Collapsed Pairs: " + cp_eval);
			         num++;
				 }
				 System.out.println();
				 writer.close();
				 
				 
				 
			} catch (Exception e) {
				e.printStackTrace();
			}
		}
			 
	}

	
}
