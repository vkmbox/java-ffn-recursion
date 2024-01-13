package vkmbox.init_distribution.recursion;

import java.util.stream.IntStream;
import lombok.AccessLevel;
import lombok.NoArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.math3.distribution.MultivariateNormalDistribution;
import org.apache.commons.math3.linear.RealVector;
import org.apache.commons.math3.stat.descriptive.moment.Mean;

@Slf4j
@NoArgsConstructor(access = AccessLevel.PRIVATE)
public class FeedForwardUtils {

    private static final int CONST_EXP_SAMPLE_SIZE=1000000;

    public static interface DoubleArrayFunction {

        /**
         * Operates on an entire array.
         *
         * @param array Array to operate on.
         * @return the result of the operation.
         */
        double evaluate(double[] array);
    }

    /*static double[] getArrayBySecondIndex(double[][] data, int index) {
        //RealMatrix matrix = new Array2DRowRealMatrix(data);
        //matrix.
        if (data.length == 0) {
            throw new IllegalArgumentException("Empty data");
        }
        if (index < 0 || index >= data[0].length) {
            throw new IllegalArgumentException("Target index out of bounds");
        }
        double[] result = new double[data.length];
        for (int ii = 0; ii < result.length; ii++) {
            result[ii] = data[ii][index];
        }
        return result;
    }
    
    static double getDotProduct(double[] one, double[] two) {
        if (one.length != two.length) {
            throw new IllegalArgumentException("Different dimensions for argument vectors");
        }
        if (one.length == 0) {
            throw new IllegalArgumentException("Empty data");
        }
        double result = 0.0;
        for (int ii = 0; ii < one.length; ii++) {
            result += one[ii]*two[ii];
        }
        return result;
    }*/
    
    //Calculation of metrics of type (4.8) and (4.36)
    static double calcGGxx(RealVector one, RealVector two, double cbn, double cwn) {
        //return cbn + cwn*getDotProduct(one, two)/one.length;
        return cbn + cwn*one.dotProduct(two)/one.getDimension();
    }
    
    static double calcSampleAverage(double[] means, double[][] covariances, DoubleArrayFunction calc) {
        Mean mean = new Mean();
        MultivariateNormalDistribution distribution = new MultivariateNormalDistribution(means, covariances);
        double[] toArray = IntStream.rangeClosed(1, CONST_EXP_SAMPLE_SIZE).parallel()
            .mapToDouble(dummy -> {
                double[] sample;
                do {
                    sample = distribution.sample();
                } while (anyNaN(sample));
                return calc.evaluate(sample);
            }).toArray();
        return mean.evaluate(toArray);
    }

    static double sumDim4(double[][][][] data) {
        if (data.length == 0 || data[0].length == 0 || data[0][0].length == 0 || data[0][0][0].length == 0) {
            return 0.0;
        }
        
        double value = 0.0;
        for (int dim1 = 0; dim1 < data.length; dim1++) {
            for (int dim2 = 0; dim2 < data[0].length; dim2++) {
                for (int dim3 = 0; dim3 < data[0][0].length; dim3++) {
                    for (int dim4 = 0; dim4 < data[0][0][0].length; dim4++) {
                        value += data[dim1][dim2][dim3][dim4];
                    }
                }
            }
        }
        return value;
    }
    
    private static boolean anyNaN(double[] array) {
        for (int ii = 0; ii < array.length; ii++) {
            if (Double.isNaN(array[ii])) {
                return true;
            }
        }
        return false;
    }
    
}
