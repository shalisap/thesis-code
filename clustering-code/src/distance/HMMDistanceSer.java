package distance ;

import weka.core.Instance ;
import weka.core.Instances ;

import java.io.File ;
import java.io.FileInputStream ;
import java.io.FileNotFoundException ;
import java.io.FileOutputStream ;
import java.io.IOException ;
import java.io.ObjectInputStream ;
import java.io.ObjectOutputStream ;
import java.util.HashMap ;
import java.util.Iterator ;

public class HMMDistanceSer implements DistanceFunction {

    private double[][] distMtx ;
    private HashMap<Instance, Integer> instIdx ;

    public static void createDistMtxFile(Instances is, File f) 
    throws FileNotFoundException, IOException {
        HashMap<Instance, Integer> instIdx = new HashMap<Instance, Integer>() ;

        for (int idx = 0 ; idx < is.numInstances(); ++idx)
            instIdx.put(is.instance(idx), idx) ;
        // for (Instance i : is) instIdx.add(i, idx++) ;

        HMMDistance hmmDist = new HMMDistance() ;
        double[][] distMtx = hmmDist.distMatrix(is) ;

        ObjectOutputStream outs = new ObjectOutputStream(new FileOutputStream(f)) ;

        outs.writeObject(instIdx) ;
        outs.writeObject(distMtx) ;

        outs.close() ;

    }

    public HMMDistanceSer(File f) 
    throws ClassNotFoundException, FileNotFoundException, IOException {
        ObjectInputStream ins = new ObjectInputStream(new FileInputStream(f)) ;

        instIdx = (HashMap<Instance, Integer>)ins.readObject() ;
        distMtx = (double[][])ins.readObject() ;

        ins.close() ;
    }

    public double distance(Instance x, Instance y) {
        Integer i = instIdx.get(x) ;
        if (i == null) throw new NullPointerException() ;
        Integer j = instIdx.get(y) ;
        if (j == null) throw new NullPointerException() ;

        return distMtx[i][j] ;
    }

    public double[][] distMatrix(Instances is) {
        return distMtx ;
    }

}
