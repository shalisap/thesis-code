package evaluation;

import static org.junit.Assert.*;

import org.junit.Test;

public class DistinguishingPairsTest {

	@Test
	public void testDistinguishingPairs() {
        int[] clusters1 = {1, 2, 2, 1, 1};
        int[] clusters2 = {2, 1, 2, 1, 1};
        DistinguishingPairs dp = new DistinguishingPairs();
        double result = dp.evaluate(clusters1, clusters2);
        assertEquals(0.4, result, 0.01);
	}

}
