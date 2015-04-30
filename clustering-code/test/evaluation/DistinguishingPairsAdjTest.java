package evaluation;

import static org.junit.Assert.*;

import org.junit.Test;

public class DistinguishingPairsAdjTest {

	@Test
	public void testDistinguishingPairs() {
        int[] clusters1 = {0, 0, 0, 1, 1, 1};
        int[] clusters2 = {0, 0, 1, 1, 2, 2};
        System.out.println("---------- Some Diff ----------");
        DistinguishingPairsAdj dpAdj = new DistinguishingPairsAdj();
        double result = dpAdj.evaluate(clusters1, clusters2);
        assertEquals(0.24, result, 0.01);
        
        double resultSym = dpAdj.evaluate(clusters2, clusters1);
        assertEquals(0.24, resultSym, 0.01);
	}
	
	@Test
	public void testDistinguishingPairsReLabel() {
        int[] clusters1 = {0, 0, 0, 1, 1, 1};
        int[] clusters2 = {1, 1, 0, 0, 3, 3};
        System.out.println("---------- Some Diff Relabeled ----------");
        DistinguishingPairsAdj dpAdj = new DistinguishingPairsAdj();
        double result = dpAdj.evaluate(clusters1, clusters2);
        assertEquals(0.24, result, 0.01);
	}
	
	@Test
	public void testDistinguishingPairsSame() {
        int[] clusters1 = {0, 0, 1, 1, 1, 1};
        int[] clusters2 = {1, 1, 0, 0, 0, 0};
        System.out.println("---------- All Same ----------");
        DistinguishingPairsAdj dpAdj = new DistinguishingPairsAdj();
        double result = dpAdj.evaluate(clusters1, clusters2);
        assertEquals(1.0, result, 0.01);
	}
	
	@Test
	public void testDistinguishingPairsAllDiff() {
        int[] clusters1 = {0, 1, 2, 0, 3, 4, 5, 1};
        int[] clusters2 = {1, 1, 0, 0, 2, 2, 2, 2};
        System.out.println("---------- All Diff ----------");
        DistinguishingPairsAdj dpAdj = new DistinguishingPairsAdj();
        double result = dpAdj.evaluate(clusters1, clusters2);
        assertEquals(-0.12, result, 0.01);
	}

}

