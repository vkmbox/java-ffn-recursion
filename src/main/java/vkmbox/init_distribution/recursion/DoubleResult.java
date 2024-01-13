package vkmbox.init_distribution.recursion;

import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutionException;
import lombok.Getter;
import lombok.AllArgsConstructor;

@AllArgsConstructor
class DoubleResult {
    
    public static enum ResultType{ VALUE, FUTURE };
    
    private final ResultType type;
    private final double value;
    private final CompletableFuture<Double> future;
    
    public double get() throws InterruptedException, ExecutionException {
        return switch(type) {
            case VALUE -> value;
            case FUTURE -> future.get();
        };
    }
    
    public static DoubleResult of(double value) {
        return new DoubleResult(ResultType.VALUE, value, null);
    }
    
    public static DoubleResult of(CompletableFuture<Double> future) {
        return new DoubleResult(ResultType.FUTURE, 0.0, future);
    }
}
