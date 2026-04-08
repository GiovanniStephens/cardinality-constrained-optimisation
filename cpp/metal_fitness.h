#pragma once
#include <vector>
#include <cstdint>

struct MetalFitnessImpl;  // Forward-declare ObjC implementation

class MetalFitnessEvaluator {
public:
    // Init: uploads centeredReturns (double*, col-major, T×M) and expectedReturns
    // to GPU as FP32. One-time cost (~50ms).
    MetalFitnessEvaluator(const double* centeredReturns, int T, int M,
                          const double* expectedReturns,
                          int minETFs, int maxETFs,
                          double riskFreeRate, double minReturn);
    ~MetalFitnessEvaluator();

    // Batch evaluate population[evalStart..evalStart+evalCount).
    // Writes evalCount doubles to outFitness.
    // Thread-safe: each call creates its own command buffer.
    // bitVecsFlat: pre-flattened uint32_t array (evalCount × wordsPerInd32).
    void evaluateBatch(const uint32_t* bitVecsFlat,
                       int evalCount, double* outFitness);

    bool isValid() const;
    int wordsPerInd32() const;  // for caller to size the flat buffer

    MetalFitnessEvaluator(const MetalFitnessEvaluator&) = delete;
    MetalFitnessEvaluator& operator=(const MetalFitnessEvaluator&) = delete;

private:
    MetalFitnessImpl* impl_;
};
