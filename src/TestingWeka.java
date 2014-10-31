import java.io.*;

import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ArffLoader;
import weka.clusterers.Cobweb;

public class TestingWeka
{
    public static void main(String[] args) throws Exception
    {
        ArffLoader loader = new ArffLoader();
        loader.setFile(new File("./data/weather.nominal.arff"));
        Instances data = loader.getStructure();

        Cobweb cw = new Cobweb(); // new instance of clusterer

        //kmeans.setOptions(options); // set options
        cw.buildClusterer(data); // build the clusterer

        Instance current;
        while ((current = loader.getNextInstance(data)) != null)
            cw.updateClusterer(current);
        cw.updateFinished();

        System.out.println(cw);

    }
}
