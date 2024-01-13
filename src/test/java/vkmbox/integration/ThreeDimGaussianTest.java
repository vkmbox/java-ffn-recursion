package vkmbox.integration;

import de.labathome.Cubature;
import java.util.HashSet;
import java.util.Set;
import java.util.stream.IntStream;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.math3.distribution.MultivariateNormalDistribution;
import org.apache.commons.math3.stat.descriptive.moment.Mean;
import org.apache.commons.lang3.tuple.Pair;
import org.junit.jupiter.api.Test;
//import org.apache.commons.math4.legacy.distribution.MultivariateNormalDistribution;

@Slf4j
public class ThreeDimGaussianTest {

    public static double[][] gaussianNd(double[][] x, Object fdata) {
        int dim = x.length;
        int nPoints = x[0].length;
        double[][] fval = new double[1][nPoints];
        double sigma = (double) fdata;
        for (int ii = 0; ii < nPoints; ++ii) {
            double sum = 0.0;
            for (int dd = 0; dd < dim; ++dd) {
                sum += x[dd][ii] * x[dd][ii];
            }
            fval[0][ii] = Math.exp(-sigma * sum);
        }
        return fval;
    }

    @Test
    void threeDimGaussianTest() {
        double[] xmin = {-100.0, -100.0, -100.0, -100.0};
        double[] xmax = {100.0, 100.0, 100.0, 100.0};
        double sigma = 0.5;
        double[][] val_err = Cubature.integrate(ThreeDimGaussianTest.class, "gaussianNd",
                xmin, xmax, //
                1.0e-4, 0, Cubature.Error.L1,
                0,
                sigma);

        log.info("Computed integral = {} +/- {}", val_err[0][0], val_err[1][0]);
    }

    @Test
    void sampleAverageTest() {
        int CONST_EXP_SAMPLE_SIZE=1000000;
        Mean mean = new Mean();
        double[] means = {0.0, 0.0};
        double[][] covariances = {{0.31659117, -0.05691738}, {-0.05691738,  0.31836267}};
        MultivariateNormalDistribution distribution = new MultivariateNormalDistribution(means, covariances);
        double[] toArray = IntStream.rangeClosed(1, CONST_EXP_SAMPLE_SIZE).parallel()
                .mapToDouble(dummy -> {
                    double[] sample;
                    do {
                        sample = distribution.sample();
                    } while (Double.isNaN(sample[0]) || Double.isNaN(sample[1]));
                    return Math.tanh(sample[0])*Math.tanh(sample[1]);
                }).toArray();
        /*double result = toArray[0];
        for (int ii = 1; ii < toArray.length; ii++) {
            double newValue = toArray[ii];
            assert newValue != Double.NaN;
            result += (newValue - result)/(ii+1);
            if (result == Double.NaN) {
                log.error("Double.NaN on iteration {}", ii);
                break;
            }
            //mean.increment(toArray[ii]);
        }*/
        //DoubleStream.of(toArray).
        double result = mean.evaluate(toArray);
        //double[] sample = distribution.sample();
        log.info("Expected value:{}", result);
        //NormalDistribution
    }
    
    @Test
    void checkPair() {
        Long aa= 146L;
        Integer bb = 641;
        Long cc= 146L;
        Integer dd = 641;
        Set<Pair> keys = new HashSet<>();
        keys.add(Pair.of(aa, bb));
        
        log.info("Result:{}", keys.contains(Pair.of(cc, dd)));
    }
}
