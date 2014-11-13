import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.stream.IntStream;

public class BestLabelings {

    /**
     * [combinations description]
     * @param  num_input [description]
     * @param  k         [description]
     * @param  startId   [description]
     * @param  branch    [description]
     * @param  numElem   [description]
     * @param  combos    [description]
     * @return           [description]
     */
    public static HashSet<ArrayList<Integer>> combinations(ArrayList<Integer> num_input, int k, int startId, Integer[] branch, int numElem,HashSet<ArrayList<Integer>> combos)
    {
        // If number of clusters the same as the number of elements in the input
        if (numElem == k)
        {
            ArrayList<Integer> one_combo = new ArrayList<Integer>();
            for(int i=0;i<branch.length;i++)
            {
                one_combo.add(branch[i]);
            }
            combos.add(one_combo);
            return combos;
        }
        for (int i = startId; i < num_input.size(); ++i)
        {
            branch[numElem++]=num_input.get(i);
            combinations(num_input, k, ++startId, branch, numElem, combos);
            --numElem;
        }
        return combos;
    }

    /**
     * [range_list description]
     * @param  min_num [description]
     * @param  max_num [description]
     * @return         [description]
     */
    public static ArrayList<Integer> range_list(int min_num, int max_num) {
        ArrayList<Integer> range = new ArrayList<Integer>();
        for (int i = min_num; i <= max_num; i++) {
            range.add(i);
        }
        return range;
    }

    /**
     *
     */
    public static ArrayList<ArrayList<String[]>> split_data(String[] input, int k){
        ArrayList<Integer> num_input = range_list(1,input.length-1);
        Integer[] branch = new Integer[k-1];
        HashSet<ArrayList<Integer>> combos =new HashSet<ArrayList<Integer>>();
        combos=combinations(num_input, k-1, 0, branch, 0, combos);

        ArrayList<ArrayList<String[]>> clusters = new ArrayList<ArrayList<String[]>>();
        // for splits in combos
        for(ArrayList<Integer> combo : combos) {
            ArrayList<String[]> single_cluster = new ArrayList<String[]>();
            int prev = 0;
            combo.add(input.length);
            for (int split : combo) {
                single_cluster.add(Arrays.copyOfRange(input, prev, split));
                prev = split;
            }
            clusters.add(single_cluster);
        }
        return clusters;
    }

    /**
     * [main description]
     * @param args[] [description]
     */
    public static void main(String args[])
    {
        String[] input = new String[] {"A","B","C","D"};
        for (int k = 1; k <= input.length + 1; k++) {
            ArrayList<ArrayList<String[]>> clusters = split_data(input, k);
            System.out.print("\nCluster" + k);
            for (ArrayList<String[]> cluster : clusters) {
                System.out.println();
                for (String[] item : cluster) {
                    System.out.print(Arrays.toString(item));
            }

        }
        }
    }
}
