package vkmbox.init_distribution.recursion;

import lombok.Getter;

@Getter
class LayerContext {
    private final int layerL;
    private final double[][] buffGGHead;
    private final double[][][][] buff4;
    private final double[][][][] buffSigmaz3;
    private final double[][][][][][] buffSigmaz4;
    //double zZg
    public LayerContext(int layerL, int trainSize) {
        this.layerL = layerL;
        buffGGHead = new double[trainSize][trainSize];
        buff4 = new double[trainSize][trainSize][trainSize][trainSize];
        buffSigmaz3 = new double[trainSize][trainSize][trainSize][trainSize];
        buffSigmaz4 = new double[trainSize][trainSize][trainSize][trainSize][trainSize][trainSize];
    }
}
