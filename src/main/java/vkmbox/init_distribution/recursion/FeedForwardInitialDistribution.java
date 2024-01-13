package vkmbox.init_distribution.recursion;

import com.google.common.collect.Sets;
import java.util.List;
import java.util.Set;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutionException;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import lombok.Getter;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.LUDecomposition;
import org.apache.commons.math3.linear.RealMatrix;
import static vkmbox.init_distribution.recursion.FeedForwardUtils.calcGGxx;
import static vkmbox.init_distribution.recursion.FeedForwardUtils.calcSampleAverage;
import static vkmbox.init_distribution.recursion.FeedForwardUtils.sumDim4;

@Slf4j
//@Builder
@RequiredArgsConstructor
public class FeedForwardInitialDistribution {
    
    //n0,nk,nl,ln=100,1000,20,9
    //nd=2
    @Getter private final int[] layerN;
    @Getter private final int trainSize;
    @Getter private final double Cb;
    @Getter private final double Cw;
    @Getter private final int constM1;
    @Getter private final int constM2;

    @Getter private RealMatrix[] GGTop;
    @Getter private RealMatrix[] GGBottom;
    @Getter private RealMatrix[] ggTop;
    @Getter private RealMatrix[] ggBottom;
    @Getter private double[][][][][] vvTop;
    @Getter private double[][][][][] vvBottom;

    private Set<Integer> indices;
    private double[] means;
    private PrimitiveDoubleFunction activation;
    
    public int getLayersNumber() {
        return layerN.length;
    }
    
    public static FeedForwardInitialDistribution calculate(Parameters params) 
            throws InterruptedException, ExecutionException {
        RealMatrix trainSet = params.getTrainSet();
        var calculator = new FeedForwardInitialDistribution(params.getWidthArray()
                , trainSet.getColumnDimension(), params.getCb(), params.getCw()
                , params.getConstM1(), params.getConstM2());
        calculator.init();
        calculator.calculateFirstLayer(trainSet);
        for (int layer = 2; layer < calculator.getLayersNumber(); layer++) {
            calculator.calculateNextLayer(layer);
        }
        return calculator;
    }
    
    private void init() {
        GGTop = new RealMatrix[getLayersNumber()];
        GGBottom = new RealMatrix[getLayersNumber()];
        ggTop = new RealMatrix[getLayersNumber()];
        ggBottom = new RealMatrix[getLayersNumber()];
        vvTop = new double[getLayersNumber()][trainSize][trainSize][trainSize][trainSize];
        vvBottom = new double[getLayersNumber()][trainSize][trainSize][trainSize][trainSize];
        indices = IntStream.range(0, trainSize)
                .boxed().collect(Collectors.toSet());
        means = new double[trainSize];
    }
    
    private void calculateFirstLayer(RealMatrix trainSet) {
        RealMatrix mGGBottom = new Array2DRowRealMatrix(trainSize, trainSize);
        for (int aa1: indices) {
            for (int aa2  = 0; aa2 <= aa1; aa2++) {
                double value = calcGGxx
                    (trainSet.getColumnVector(aa1), trainSet.getColumnVector(aa2), Cb, Cw);
                mGGBottom.setEntry(aa1, aa2, value);
                mGGBottom.setEntry(aa2, aa1, value);
            }
        }
        ggBottom[0] = GGBottom[0] = mGGBottom;
        ggTop[0] = GGTop[0] = new LUDecomposition(mGGBottom).getSolver().getInverse();
    }

    private void calculateNextLayer(int layerToCalculate) throws InterruptedException, ExecutionException {
        int layerL = layerToCalculate-1;
        int layerNextL = layerToCalculate;

        log.info("Starting calculation for layer {}",layerNextL);
        RealMatrix mggBottom = ggBottom[layerL] //, mGGBottom = GGBottom[layerL]
                , mGGBottomNext = new Array2DRowRealMatrix(trainSize, trainSize);
        LayerContext context = new LayerContext(layerL, trainSize);
        //double zZg = Math.pow(2*Math.PI, trainSize/2)*Math.pow(new LUDecomposition(mggBottom).getDeterminant(), 0.5);

        log.debug("Calculating G-coefficients with bottom indices for layer {}",layerNextL);
        for (int aa1: indices) {
            for (int aa2 = 0; aa2 <= aa1; aa2++) {
                log.debug("Calculating G-coefficient for aa: {},{}", aa1, aa2);
                DoubleResult integralGG = calcIntegralGG(aa1, aa2, context);
                
                double [][][][] buffTail = new double[trainSize][trainSize][trainSize][trainSize];
                for (List<Integer> bbIdx : Sets.cartesianProduct(List.of(indices, indices, indices, indices))) {
                    int bb1 = bbIdx.get(0), bb2 = bbIdx.get(1), bb3 = bbIdx.get(2), bb4 = bbIdx.get(3);
                    double valVvTop = vvTop[layerL][bb1][bb2][bb3][bb4];
                    if (valVvTop != 0.0) {//check if on previous iteration vv was calculated, 'd be true from 2nd layer
                        log.debug("Calculating G-term for bb: {},{},{},{}", bb1,bb2,bb3,bb4);
                        DoubleResult integralSigmaz3 = calcIntegralSigmaz3(aa1,aa2,bb1,bb2, context);
                        DoubleResult integralSigmaz4 = calcIntegralSigmaz4(aa1,aa2,bb1,bb2,bb3,bb4, context);
                        buffTail[bb1][bb2][bb3][bb4] = valVvTop*
                            (integralSigmaz4.get()+2*constM1*mggBottom.getEntry(bb3, bb4)*integralSigmaz3.get()
                                -2*integralGG.get()*mggBottom.getEntry(bb1, bb3)*mggBottom.getEntry(bb2, bb4));
                    }
                }
                double value = integralGG.get()+sumDim4(buffTail)/8;
                mGGBottomNext.setEntry(aa1, aa2, value);
                mGGBottomNext.setEntry(aa2, aa1, value);
//                matrix_GG_bottom[layer_next_zb, aa_1, aa_2] = matrix_GG_bottom[layer_next_zb, aa_2, aa_1] = GG_bottom
            }
        }
        GGBottom[layerNextL] = mGGBottomNext;
        RealMatrix mGGTopNext = GGTop[layerNextL] = new LUDecomposition(mGGBottomNext).getSolver().getInverse();
        
        log.debug("Calculating v-coefficients with bottom indices for layer {}", layerNextL);
        for (List<Integer> aaIdx : Sets.cartesianProduct(List.of(indices, indices, indices, indices))) {
            int aa1 = aaIdx.get(0), aa2 = aaIdx.get(1), aa3 = aaIdx.get(2), aa4 = aaIdx.get(3);
            log.debug("Calculating vv_coefficient for aa: {},{},{},{}", aa1,aa2,aa3,aa4);
            DoubleResult integral4 = calcIntegral4(aa1,aa2,aa3,aa4, context);
            DoubleResult integral2_12 = calcIntegralGG(aa1,aa2, context);
            DoubleResult integral2_34 = calcIntegralGG(aa3,aa4, context);

            double [][][][] buffTail = new double[trainSize][trainSize][trainSize][trainSize];
            for (List<Integer> bbIdx : Sets.cartesianProduct(List.of(indices, indices, indices, indices))) {
                int bb1 = bbIdx.get(0), bb2 = bbIdx.get(1), bb3 = bbIdx.get(2), bb4 = bbIdx.get(3);
                double vvTopValue = vvTop[layerL][bb1][bb2][bb3][bb4];
                if (vvTopValue != 0.0) {
                    DoubleResult integralSigmaz3_12 = calcIntegralSigmaz3(aa1,aa2,bb1,bb2, context);
                    DoubleResult integralSigmaz3_34 = calcIntegralSigmaz3(aa3,aa4,bb3,bb4, context);
                    buffTail[bb1][bb2][bb3][bb4]=vvTopValue*integralSigmaz3_12.get()*integralSigmaz3_34.get();
                }
            }
            vvBottom[layerNextL][aa1][aa2][aa3][aa4] = (integral4.get()-integral2_12.get()*integral2_34.get())/layerN[layerL] + sumDim4(buffTail)/4;
        }

        log.debug("Calculating g-coefficients with top indices for layer {}", layerNextL);
        RealMatrix mggTopNext = new Array2DRowRealMatrix(trainSize, trainSize);
        for (int aa1: indices) {
            for (int aa2 = 0; aa2 <= aa1; aa2++) {
                log.debug("Calculating g-coefficient for aa: {},{}", aa1,aa2);
                double [][][][] buffTail = new double[trainSize][trainSize][trainSize][trainSize];
                for (List<Integer> bbIdx : Sets.cartesianProduct(List.of(indices, indices, indices, indices))) {
                    int bb1 = bbIdx.get(0), bb2 = bbIdx.get(1), bb3 = bbIdx.get(2), bb4 = bbIdx.get(3);
                    buffTail[bb1][bb2][bb3][bb4] = vvBottom[layerNextL][bb1][bb2][bb3][bb4]*mGGTopNext.getEntry(aa1,bb1)*
                        (mGGTopNext.getEntry(bb2,bb3)*mGGTopNext.getEntry(bb4,aa2)+constM2*mGGTopNext.getEntry(bb2,aa2)*mGGTopNext.getEntry(bb3,bb4)/2);
                }
                double value = mGGTopNext.getEntry(aa1,aa2) + sumDim4(buffTail);
                mggTopNext.setEntry(aa1, aa2, value);
                mggTopNext.setEntry(aa2, aa1, value);
            }
        }
        ggTop[layerNextL] = mggTopNext;
        ggBottom[layerNextL] = new LUDecomposition(mggTopNext).getSolver().getInverse();
        
        log.debug("Calculating v-coefficients with top indices for layer {}", layerNextL);
        for (List<Integer> aaIdx : Sets.cartesianProduct(List.of(indices, indices, indices, indices))) {
            int aa1 = aaIdx.get(0), aa2 = aaIdx.get(1), aa3 = aaIdx.get(2), aa4 = aaIdx.get(3);
            log.debug("Calculating vv_coefficient for aa: {},{},{},{}", aa1,aa2,aa3,aa4);
            double [][][][] buffTail = new double[trainSize][trainSize][trainSize][trainSize];
            for (List<Integer> bbIdx : Sets.cartesianProduct(List.of(indices, indices, indices, indices))) {
                int bb1 = bbIdx.get(0), bb2 = bbIdx.get(1), bb3 = bbIdx.get(2), bb4 = bbIdx.get(3);            
                buffTail[bb1][bb2][bb3][bb4]=mGGTopNext.getEntry(aa1,bb1)*mGGTopNext.getEntry(aa2,bb2)*mGGTopNext.getEntry(aa3,bb3)*
                    mGGTopNext.getEntry(aa4,bb4)*vvBottom[layerNextL][bb1][bb2][bb3][bb4];
            }
            vvTop[layerNextL][aa1][aa2][aa3][aa4] = sumDim4(buffTail);
        }
    }

    private DoubleResult calcIntegralGG(int aa1, int aa2, LayerContext context) {
        double[][] buffGGHead = context.getBuffGGHead();
        double calculated = buffGGHead[aa1][aa2];
        if (calculated != 0.0) {
            return DoubleResult.of(calculated);
        }

        CompletableFuture<Double> future = CompletableFuture.supplyAsync(() -> {
            double value = calcSampleAverage(means, ggTop[context.getLayerL()].getData()
                    , (args) -> activation.apply(args[aa1])*activation.apply(args[aa2]));
            buffGGHead[aa1][aa2] = buffGGHead[aa2][aa1] = value;
            return value;
        });
        return DoubleResult.of(future);
    }

    private DoubleResult calcIntegralSigmaz4(int aa1, int aa2
            , int bb1, int bb2, int bb3, int bb4, LayerContext context) {
        double[][][][][][] buffSigmaz4 = context.getBuffSigmaz4();
        double calculated = buffSigmaz4[aa1][aa2][bb1][bb2][bb3][bb4];
        if (calculated != 0.0) {
            return DoubleResult.of(calculated);
        }

        CompletableFuture<Double> future = CompletableFuture.supplyAsync(() -> {
            int layerL = context.getLayerL();
            double value = calcSampleAverage(means, ggTop[layerL].getData()
                    , (args) -> activation.apply(args[aa1])*activation.apply(args[aa2])
                        *zJointBottom(context.getLayerL(), bb1, bb2, args)*zJointBottom(layerL, bb3, bb4, args));
            buffSigmaz4[aa1][aa2][bb1][bb2][bb3][bb4] = buffSigmaz4[aa1][aa2][bb2][bb1][bb3][bb4] =
            buffSigmaz4[aa1][aa2][bb1][bb2][bb4][bb3] = buffSigmaz4[aa1][aa2][bb2][bb1][bb4][bb3] =
            buffSigmaz4[aa1][aa2][bb3][bb4][bb1][bb2] = buffSigmaz4[aa1][aa2][bb3][bb4][bb2][bb1] =
            buffSigmaz4[aa1][aa2][bb4][bb3][bb1][bb2] = buffSigmaz4[aa1][aa2][bb4][bb3][bb2][bb1] =
            buffSigmaz4[aa2][aa1][bb1][bb2][bb3][bb4] = buffSigmaz4[aa2][aa1][bb2][bb1][bb3][bb4] =
            buffSigmaz4[aa2][aa1][bb1][bb2][bb4][bb3] = buffSigmaz4[aa2][aa1][bb2][bb1][bb4][bb3] =
            buffSigmaz4[aa2][aa1][bb3][bb4][bb1][bb2] = buffSigmaz4[aa2][aa1][bb3][bb4][bb2][bb1] =
            buffSigmaz4[aa2][aa1][bb4][bb3][bb1][bb2] = buffSigmaz4[aa2][aa1][bb4][bb3][bb2][bb1] = value;
            return value;
        });
        return DoubleResult.of(future);
    }

    private DoubleResult calcIntegralSigmaz3(int aa1, int aa2, int bb1, int bb2, LayerContext context) {
        double[][][][] buffSigmaz3 = context.getBuffSigmaz3();
        double calculated = buffSigmaz3[aa1][aa2][bb1][bb2];
        if (calculated != 0.0) {
            return DoubleResult.of(calculated);
        }

        CompletableFuture<Double> future = CompletableFuture.supplyAsync(() -> {
            double value = calcSampleAverage(means, ggTop[context.getLayerL()].getData()
                    , (args) -> activation.apply(args[aa1])*activation.apply(args[aa2])*zJointBottom(context.getLayerL(), bb1, bb2, args));
            buffSigmaz3[aa1][aa2][bb1][bb2] = buffSigmaz3[aa1][aa2][bb2][bb1] =
            buffSigmaz3[aa2][aa1][bb1][bb2] = buffSigmaz3[aa2][aa1][bb2][bb1] = value;
            return value;
        });
        return DoubleResult.of(future);
    }
    
    //TODO: possible result-buffer for 4! aa1-aa4 permutations
    private DoubleResult calcIntegral4(int aa1, int aa2, int aa3, int aa4, LayerContext context) {
        double[][][][] buff4 = context.getBuff4();
        double calculated = buff4[aa1][aa2][aa3][aa4];
        if (calculated != 0.0) {
            return DoubleResult.of(calculated);
        }        
        
        CompletableFuture<Double> future = CompletableFuture.supplyAsync(() -> {
            double value = calcSampleAverage(means, ggTop[context.getLayerL()].getData()
                    , (args) -> activation.apply(args[aa1])*activation.apply(args[aa2])*activation.apply(args[aa3])*activation.apply(args[aa4]));
            buff4[aa1][aa2][aa3][aa4] = buff4[aa1][aa2][aa4][aa3] =
            buff4[aa1][aa3][aa2][aa4] = buff4[aa1][aa3][aa4][aa2] =
            buff4[aa1][aa4][aa2][aa3] = buff4[aa1][aa4][aa3][aa2] =
            buff4[aa2][aa1][aa3][aa4] = buff4[aa2][aa1][aa4][aa3] =
            buff4[aa2][aa3][aa1][aa4] = buff4[aa2][aa3][aa4][aa1] =
            buff4[aa2][aa4][aa1][aa3] = buff4[aa2][aa4][aa3][aa1] =
            buff4[aa3][aa1][aa2][aa4] = buff4[aa3][aa1][aa4][aa2] =
            buff4[aa3][aa2][aa1][aa4] = buff4[aa3][aa2][aa4][aa1] =
            buff4[aa3][aa4][aa1][aa2] = buff4[aa3][aa4][aa2][aa1] =
            buff4[aa4][aa1][aa2][aa3] = buff4[aa4][aa1][aa3][aa2] =
            buff4[aa4][aa2][aa1][aa3] = buff4[aa4][aa2][aa3][aa1] =
            buff4[aa4][aa3][aa1][aa2] = buff4[aa4][aa3][aa2][aa1] = value;

            return value;
        });
        return DoubleResult.of(future);
    }

    private double zJointBottom(int layerL, int bb1, int bb2, double[] sample) {
        return sample[bb1]*sample[bb2]-ggBottom[layerL].getEntry(bb1, bb2);
    }
}
