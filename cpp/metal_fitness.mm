#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include "metal_fitness.h"
#include <iostream>
#include <cmath>
#include <vector>
#include <cstring>

// ─── Metal Shader ───────────────────────────────────────────────────────────────
//
// Architecture: each threadgroup evaluates ONE portfolio. Multiple threads within
// the group parallelize over the TIME dimension (T=1260 rows). Thread 0 extracts
// selected ETF indices via bit-scan, then all threads accumulate column values
// for their time slice and compute partial squared norms. A tree reduction
// combines partial norms. Thread 0 computes the final Sharpe ratio.
//
// This gives coalesced memory access (adjacent threads read adjacent time indices)
// and full SIMD utilization (THREADS_PER_GROUP threads per group).

static const char* metalShaderSource = R"(
#include <metal_stdlib>
using namespace metal;

struct Params {
    uint T;            // number of return observations
    uint M;            // number of instruments
    uint wordsPerInd;  // uint32 words per individual's bit vector
    int  minETFs;
    int  maxETFs;
    float riskFreeRate;
    float minReturn;   // negative = no constraint
};

kernel void fitness_kernel(
    device const float*  centeredReturns [[buffer(0)]],  // T×M col-major
    device const float*  expectedReturns [[buffer(1)]],  // length M
    device const uint*   bitVectors      [[buffer(2)]],  // evalCount × wordsPerInd
    device float*        outFitness      [[buffer(3)]],  // evalCount
    constant Params&     params          [[buffer(4)]],
    uint gid    [[threadgroup_position_in_grid]],
    uint tid    [[thread_index_in_threadgroup]],
    uint tcount [[threads_per_threadgroup]])
{
    uint T = params.T;
    uint M = params.M;
    uint wordsPerInd = params.wordsPerInd;
    device const uint* myBits = bitVectors + gid * wordsPerInd;

    // Phase 1: Thread 0 extracts selected ETF indices and computes expected return sum
    threadgroup uint selectedIndices[MAX_SELECTED];
    threadgroup int numSelected;
    threadgroup float sumER;

    if (tid == 0) {
        numSelected = 0;
        sumER = 0.0f;
        for (uint w = 0; w < wordsPerInd; ++w) {
            uint bits = myBits[w];
            while (bits) {
                int bit = ctz(bits);
                uint idx = w * 32 + bit;
                if (idx < M && numSelected < MAX_SELECTED) {
                    selectedIndices[numSelected] = idx;
                    sumER += expectedReturns[idx];
                    numSelected++;
                }
                bits &= (bits - 1);
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    int n = numSelected;

    // Cardinality check
    if (n < params.minETFs || n > params.maxETFs) {
        if (tid == 0) outFitness[gid] = -1e4f;
        return;
    }

    float portfolioReturn = sumER / float(n);
    if (params.minReturn >= 0.0f && portfolioReturn < params.minReturn) {
        if (tid == 0) outFitness[gid] = -1e4f;
        return;
    }

    // Phase 2: Each thread accumulates portfolio values for its time slice
    // and computes partial squared norm. Adjacent threads access adjacent
    // time indices → coalesced memory access.
    float partialSqNorm = 0.0f;

    for (uint t = tid; t < T; t += tcount) {
        float portVal = 0.0f;
        for (int s = 0; s < n; ++s) {
            portVal += centeredReturns[selectedIndices[s] * T + t];
        }
        partialSqNorm += portVal * portVal;
    }

    // Phase 3: Tree reduction of partial squared norms across threadgroup
    threadgroup float partials[THREADS_PER_GROUP];
    partials[tid] = partialSqNorm;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = tcount / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            partials[tid] += partials[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Thread 0 computes final Sharpe ratio
    if (tid == 0) {
        float sqNorm = partials[0];
        float portfolioVar = sqNorm / (float(n) * float(n) * float(T - 1));
        if (portfolioVar <= 0.0f) {
            outFitness[gid] = -1e4f;
        } else {
            float portfolioRisk = sqrt(portfolioVar) * sqrt(252.0f);
            outFitness[gid] = (portfolioReturn - params.riskFreeRate) / portfolioRisk;
        }
    }
}
)";

// ─── Constants ──────────────────────────────────────────────────────────────────

static constexpr int THREADS_PER_GROUP = 64;  // 2 SIMD groups per threadgroup
static constexpr int MAX_SELECTED = 64;       // max ETFs per portfolio

// Tree reduction in the shader requires a power-of-2 threadgroup size.
static_assert((THREADS_PER_GROUP & (THREADS_PER_GROUP - 1)) == 0,
              "THREADS_PER_GROUP must be a power of 2 for correct tree reduction");

// ─── Params struct (must match shader layout) ───────────────────────────────────

struct MetalParams {
    uint32_t T;
    uint32_t M;
    uint32_t wordsPerInd;
    int32_t  minETFs;
    int32_t  maxETFs;
    float    riskFreeRate;
    float    minReturn;
};

// ─── Implementation ─────────────────────────────────────────────────────────────

struct MetalFitnessImpl {
    id<MTLDevice>              device;
    id<MTLCommandQueue>        queue;
    id<MTLComputePipelineState> pipeline;
    id<MTLBuffer>              centeredReturnsBuf;  // float, T×M col-major
    id<MTLBuffer>              expectedReturnsBuf;  // float, M
    // NOTE: bitVecBuf and outBuf are allocated per-call in evaluateBatch
    // to maintain thread safety across islands.
    int T;
    int M;
    int wpi32;  // words per individual (uint32)
    int minETFs;
    int maxETFs;
    float riskFreeRate;
    float minReturn;
    bool valid;
};

// ─── Constructor ────────────────────────────────────────────────────────────────

MetalFitnessEvaluator::MetalFitnessEvaluator(
    const double* centeredReturns, int T, int M,
    const double* expectedReturns,
    int minETFs, int maxETFs,
    double riskFreeRate, double minReturn)
{
    impl_ = new MetalFitnessImpl();
    impl_->valid = false;
    impl_->T = T;
    impl_->M = M;
    impl_->minETFs = minETFs;
    impl_->maxETFs = maxETFs;
    impl_->riskFreeRate = static_cast<float>(riskFreeRate);
    impl_->minReturn = static_cast<float>(minReturn);

    // uint32 words per individual: each uint64 word becomes 2 uint32 words
    int numWords64 = (M + 63) / 64;
    impl_->wpi32 = numWords64 * 2;

    if (maxETFs > MAX_SELECTED) {
        std::cerr << "Metal: maxETFs=" << maxETFs << " exceeds GPU limit of "
                  << MAX_SELECTED << ", falling back to CPU." << std::endl;
        return;
    }

    @autoreleasepool {
        impl_->device = MTLCreateSystemDefaultDevice();
        if (!impl_->device) {
            std::cerr << "Metal: No GPU device found." << std::endl;
            return;
        }

        // Compile shader with preprocessor defines
        NSError* error = nil;
        MTLCompileOptions* options = [[MTLCompileOptions alloc] init];
        options.preprocessorMacros = @{
            @"THREADS_PER_GROUP": @(THREADS_PER_GROUP),
            @"MAX_SELECTED": @(MAX_SELECTED)
        };

        NSString* source = [NSString stringWithUTF8String:metalShaderSource];
        id<MTLLibrary> library = [impl_->device newLibraryWithSource:source
                                                             options:options
                                                               error:&error];
        if (!library) {
            std::cerr << "Metal: Shader compilation failed: "
                      << [[error localizedDescription] UTF8String] << std::endl;
            return;
        }

        id<MTLFunction> function = [library newFunctionWithName:@"fitness_kernel"];
        if (!function) {
            std::cerr << "Metal: Could not find fitness_kernel function." << std::endl;
            return;
        }

        impl_->pipeline = [impl_->device newComputePipelineStateWithFunction:function
                                                                       error:&error];
        if (!impl_->pipeline) {
            std::cerr << "Metal: Pipeline creation failed: "
                      << [[error localizedDescription] UTF8String] << std::endl;
            return;
        }

        impl_->queue = [impl_->device newCommandQueue];
        if (!impl_->queue) {
            std::cerr << "Metal: Command queue creation failed." << std::endl;
            return;
        }

        // Convert centeredReturns (double*, col-major T×M) to float and upload
        size_t crSize = static_cast<size_t>(T) * M;
        std::vector<float> crFloat(crSize);
        for (size_t i = 0; i < crSize; ++i)
            crFloat[i] = static_cast<float>(centeredReturns[i]);

        impl_->centeredReturnsBuf = [impl_->device
            newBufferWithBytes:crFloat.data()
                        length:crSize * sizeof(float)
                       options:MTLResourceStorageModeShared];

        // Convert expectedReturns (double*, length M) to float and upload
        std::vector<float> erFloat(M);
        for (int i = 0; i < M; ++i)
            erFloat[i] = static_cast<float>(expectedReturns[i]);

        impl_->expectedReturnsBuf = [impl_->device
            newBufferWithBytes:erFloat.data()
                        length:M * sizeof(float)
                       options:MTLResourceStorageModeShared];

        if (!impl_->centeredReturnsBuf || !impl_->expectedReturnsBuf) {
            std::cerr << "Metal: Buffer allocation failed." << std::endl;
            return;
        }

        impl_->valid = true;
        std::cerr << "Metal: GPU fitness evaluator initialized (T=" << T
                  << ", M=" << M << ", " << impl_->wpi32 << " words/ind, "
                  << THREADS_PER_GROUP << " threads/group)." << std::endl;
    }
}

// ─── Destructor ─────────────────────────────────────────────────────────────────

MetalFitnessEvaluator::~MetalFitnessEvaluator() {
    delete impl_;
}

// ─── evaluateBatch ──────────────────────────────────────────────────────────────

void MetalFitnessEvaluator::evaluateBatch(
    const uint32_t* bitVecsFlat, int evalCount, double* outFitness)
{
    if (!impl_->valid || evalCount <= 0) {
        for (int i = 0; i < evalCount; ++i)
            outFitness[i] = -1e4;
        return;
    }

    @autoreleasepool {
        size_t bitBufSize = static_cast<size_t>(evalCount) * impl_->wpi32 * sizeof(uint32_t);
        size_t outBufSize = static_cast<size_t>(evalCount) * sizeof(float);

        // Allocate shared-mode buffer + memcpy (avoids the implicit copy in
        // newBufferWithBytes which can be slower for large buffers).
        id<MTLBuffer> bitVecBuf = [impl_->device
            newBufferWithLength:bitBufSize
                        options:MTLResourceStorageModeShared];
        id<MTLBuffer> outBuf = [impl_->device
            newBufferWithLength:outBufSize
                        options:MTLResourceStorageModeShared];
        if (!bitVecBuf || !outBuf) {
            std::cerr << "Metal: Buffer allocation failed in evaluateBatch." << std::endl;
            for (int i = 0; i < evalCount; ++i) outFitness[i] = -1e4;
            return;
        }
        memcpy([bitVecBuf contents], bitVecsFlat, bitBufSize);

        // Set up params
        MetalParams params;
        params.T = static_cast<uint32_t>(impl_->T);
        params.M = static_cast<uint32_t>(impl_->M);
        params.wordsPerInd = static_cast<uint32_t>(impl_->wpi32);
        params.minETFs = impl_->minETFs;
        params.maxETFs = impl_->maxETFs;
        params.riskFreeRate = impl_->riskFreeRate;
        params.minReturn = impl_->minReturn;

        // Encode and dispatch
        id<MTLCommandBuffer> cmdBuf = [impl_->queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [cmdBuf computeCommandEncoder];
        if (!cmdBuf || !encoder) {
            std::cerr << "Metal: Command buffer/encoder creation failed." << std::endl;
            for (int i = 0; i < evalCount; ++i) outFitness[i] = -1e4;
            return;
        }

        [encoder setComputePipelineState:impl_->pipeline];
        [encoder setBuffer:impl_->centeredReturnsBuf offset:0 atIndex:0];
        [encoder setBuffer:impl_->expectedReturnsBuf offset:0 atIndex:1];
        [encoder setBuffer:bitVecBuf                  offset:0 atIndex:2];
        [encoder setBuffer:outBuf                    offset:0 atIndex:3];
        [encoder setBytes:&params length:sizeof(MetalParams) atIndex:4];

        MTLSize threadgroupSize = MTLSizeMake(THREADS_PER_GROUP, 1, 1);
        MTLSize gridSize = MTLSizeMake(static_cast<NSUInteger>(evalCount), 1, 1);
        [encoder dispatchThreadgroups:gridSize threadsPerThreadgroup:threadgroupSize];

        [encoder endEncoding];
        [cmdBuf commit];
        [cmdBuf waitUntilCompleted];

        if ([cmdBuf status] == MTLCommandBufferStatusError) {
            std::cerr << "Metal: Command buffer error: "
                      << [[[cmdBuf error] localizedDescription] UTF8String] << std::endl;
            for (int i = 0; i < evalCount; ++i) outFitness[i] = -1e4;
            return;
        }

        // Read results back as float, convert to double
        const float* results = static_cast<const float*>([outBuf contents]);
        for (int i = 0; i < evalCount; ++i)
            outFitness[i] = static_cast<double>(results[i]);
    }
}

// ─── Accessors ──────────────────────────────────────────────────────────────────

bool MetalFitnessEvaluator::isValid() const {
    return impl_ && impl_->valid;
}

int MetalFitnessEvaluator::wordsPerInd32() const {
    return impl_ ? impl_->wpi32 : 0;
}
