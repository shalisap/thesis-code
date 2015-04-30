package clustering;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import com.opencsv.CSVWriter;

/**
 * With a directory of 
 * 
 * @author Shalisa Pattarawuttiwong
 */
public class ToCSV {
	public static void main(String[] args) throws Exception{
		String output = args[0];
		//String output = "clustering_assessments.csv";
		CSVWriter writer = new CSVWriter(new FileWriter(output));
		
		int minK = 2;
		int maxK = 10;
		
		// header - number clusters
		String[] numClus = new String[29];
		String[] header = new String[29];
		numClus[0] = "";
		numClus[1] = "Number of clusters";
		
		header[0] = "Clustering Algorithm";
		header[1] = "Distance Function";
		int numCol = 2;
		
		for (int i = minK; i <= maxK; i++) {
			String k = Integer.toString(i);
			numClus[numCol] = k;
			numClus[numCol+1] = k;
			numClus[numCol+2] = k;
			
			header[numCol] = "Rand Index";
			header[numCol+1] = "Adj Rand Index";
			header[numCol+2] = "Collapsed Pairs";
			numCol += 3;
		}

		// Find .log files in directory
		String path = "."; 

		String filesString;
		File folder = new File(path);
		File[] listOfFiles = folder.listFiles(); 
		List<String> files = new ArrayList<String>();
		
		for (int i = 0; i < listOfFiles.length; i++) 
		{

			if (listOfFiles[i].isFile()) 
			{
				filesString = listOfFiles[i].getName();
				if (filesString.endsWith(".log"))
				{
					files.add(path + "/" + filesString);
				}
			}
		}
		
		String[][] data = new String[files.size()][29];
		//read from
		for (int i = 0; i < files.size(); i++) {
		try {
			File file = new File(files.get(i));
			FileReader fileReader = new FileReader(file);
			BufferedReader bufferedReader = new BufferedReader(fileReader);
			String line;
			
			String alg = "";
			String dist = "";
			String indices = "";
			while ((line = bufferedReader.readLine()) != null) {
				if (line.contains("Clustering Algorithm")) {
					String[] split = line.split(" ");
					alg = split[split.length-1];
				} else if (line.contains("Distance Measure")){
					String[] split = line.split(" ");
					dist = split[split.length-1];
				} else if (alg.equalsIgnoreCase("hierarchical") && 
						line.contains("Agglomeration")) {
					String[] split = line.split(" ");
					if (line.contains("weighted average")) {
						alg = alg + " " + split[split.length-2] + 
								" " + split[split.length-1];
					} else {
						alg = alg + " " + split[split.length-1];
					}
				} else if (line.contains("Rand Index") || 
						line.contains("Collapsed")) {
					String[] split = line.split(" ");
					indices = indices + "," + split[split.length-1];
				}
			}
			data[i] = (alg + "," + dist + indices).split(",");
			
			fileReader.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
		}
		
		// write to file
		writer.writeNext(numClus);
		writer.writeNext(header);
		
		for (String[] d: data) {
		writer.writeNext(d);
		}
		writer.close();
		
		System.out.println("Export to .csv file [" + output + "] complete");
	}
}
