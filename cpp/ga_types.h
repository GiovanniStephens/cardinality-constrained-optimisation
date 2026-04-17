#pragma once

#include <Eigen/Dense>
#include <vector>
#include <string>
#include <random>
#include <algorithm>
#include <cmath>
#include <mutex>
#include <limits>
#include <cassert>
#include <numeric>

// ─── Configuration ─────────────────────────────────────────────────────────────

struct Config {
    std::string mode = "ga";  // "ga" or "mc"
    std::string data_path = "data/ETF_Prices.csv";
    int pop_size = 1000;
    int num_generations = 50;
    int min_etfs = 3;
    int max_etfs = 15;
    double risk_free_rate = 0.0;
    int seed = -1;           // -1 = use random_device
    int num_islands = -1;    // -1 = hardware_concurrency
    double time_budget = -1; // seconds, -1 = no limit
    double min_return = -1;  // -1 = no constraint
    int num_elites = 10;
    int migration_interval = 10;
    double migration_rate = 0.1;
    int top_k = 100;
    double missing_threshold = 0.02; // fraction of rows allowed to be NaN
    int mc_log_interval = 5000;      // MC: log every N trials per thread
    bool binary_input = false;       // if true, read binary format instead of CSV
    bool use_svd = false;            // if true, use truncated SVD for fitness
    int svd_components = 200;        // number of SVD components to keep
    bool use_gpu = false;            // if true, use Metal GPU for fitness
    // Adaptive mutation: linear decay from mutation_initial to mutation_final
    double mutation_initial = 0.02;  // 2% early exploration
    double mutation_final   = 0.005; // 0.5% late exploitation (higher floor than before)
    // Stagnation restart: reinitialise island after N gens without improvement
    int stagnation_restart = 200;    // 0 = disable restart
};

// Declaration only — implemented in optimisation.cpp (called once from main)
Config parse_args(int argc, char* argv[]);

// ─── Bitwise individual representation (Phase 2) ──────────────────────────────

using BitIndividual = std::vector<uint64_t>;
static constexpr int BITS_PER_WORD = 64;

inline int numWords(int numGenes) {
    return (numGenes + BITS_PER_WORD - 1) / BITS_PER_WORD;
}

inline bool getBit(const BitIndividual& ind, int pos) {
    assert(pos >= 0 && pos < static_cast<int>(ind.size() * BITS_PER_WORD));
    return (ind[pos / BITS_PER_WORD] >> (pos % BITS_PER_WORD)) & 1ULL;
}

inline void setBit(BitIndividual& ind, int pos) {
    assert(pos >= 0 && pos < static_cast<int>(ind.size() * BITS_PER_WORD));
    ind[pos / BITS_PER_WORD] |= (1ULL << (pos % BITS_PER_WORD));
}

inline void clearBit(BitIndividual& ind, int pos) {
    ind[pos / BITS_PER_WORD] &= ~(1ULL << (pos % BITS_PER_WORD));
}

inline void flipBit(BitIndividual& ind, int pos) {
    ind[pos / BITS_PER_WORD] ^= (1ULL << (pos % BITS_PER_WORD));
}

inline int popcount(const BitIndividual& ind) {
    int count = 0;
    for (auto w : ind) count += __builtin_popcountll(w);
    return count;
}

// ─── Lightweight RNG for parallel GA operators ─────────────────────────────────
// SplitMix64: 64-bit state, passes BigCrush. Used per-offspring in dispatch_apply
// to avoid contention on the island's mt19937.

struct SplitMix64 {
    using result_type = uint64_t;
    uint64_t state;

    explicit SplitMix64(uint64_t s) : state(s) {}

    static constexpr result_type min() { return 0; }
    static constexpr result_type max() { return UINT64_MAX; }

    result_type operator()() {
        state += 0x9e3779b97f4a7c15ULL;
        uint64_t z = state;
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
        return z ^ (z >> 31);
    }
};

// ─── Cardinality repair ────────────────────────────────────────────────────────

template<typename RNG>
inline void repairCardinality(BitIndividual& ind, int numGenes,
                       int minETFs, int maxETFs, RNG& rng,
                       std::vector<int>& buf) {
    int current = popcount(ind);
    std::uniform_int_distribution<int> targetDist(minETFs, maxETFs);
    int targetCount = targetDist(rng);

    if (current == targetCount) return;

    buf.clear();
    if (current > targetCount) {
        // Collect set bit indices, shuffle, clear excess
        for (size_t w = 0; w < ind.size(); ++w) {
            uint64_t bits = ind[w];
            while (bits) {
                int bit = __builtin_ctzll(bits);
                int idx = static_cast<int>(w) * BITS_PER_WORD + bit;
                if (idx < numGenes) buf.push_back(idx);
                bits &= bits - 1;
            }
        }
        std::shuffle(buf.begin(), buf.end(), rng);
        for (int i = 0; i < current - targetCount; ++i)
            clearBit(ind, buf[i]);
    } else {
        // Collect clear bit indices, shuffle, set deficit
        for (int i = 0; i < numGenes; ++i) {
            if (!getBit(ind, i)) buf.push_back(i);
        }
        std::shuffle(buf.begin(), buf.end(), rng);
        for (int i = 0; i < targetCount - current; ++i)
            setBit(ind, buf[i]);
    }
}

// ─── GA operators (bitwise) ────────────────────────────────────────────────────

inline std::vector<BitIndividual> initializePopulation(int size, int numGenes,
                                                  int minETFs, int maxETFs,
                                                  std::mt19937& rng) {
    int nw = numWords(numGenes);
    double prob = static_cast<double>(maxETFs) / numGenes;
    std::bernoulli_distribution dist(prob);
    std::vector<BitIndividual> population(size, BitIndividual(nw, 0));
    std::vector<int> repairBuf;
    repairBuf.reserve(numGenes);

    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < numGenes; ++j) {
            if (dist(rng)) setBit(population[i], j);
        }
        repairCardinality(population[i], numGenes, minETFs, maxETFs, rng, repairBuf);
    }
    return population;
}

// Uniform crossover: for each 64-bit word, generate a random mask and blend
template<typename RNG>
inline BitIndividual crossoverOne(const BitIndividual& p1, const BitIndividual& p2,
                            RNG& rng) {
    BitIndividual child(p1.size());
    for (size_t w = 0; w < p1.size(); ++w) {
        uint64_t mask;
        if constexpr (sizeof(typename RNG::result_type) >= 8) {
            // 64-bit RNG (e.g. SplitMix64): single call suffices
            mask = rng();
        } else {
            // 32-bit RNG (e.g. mt19937): combine two calls
            mask = static_cast<uint64_t>(rng()) |
                   (static_cast<uint64_t>(rng()) << 32);
        }
        child[w] = (p1[w] & mask) | (p2[w] & ~mask);
    }
    return child;
}

// Phase 3: Poisson mutation — O(1) expected per individual instead of O(numGenes)
template<typename RNG>
inline void mutateOne(BitIndividual& ind, double mutationRate, int numGenes,
               RNG& rng) {
    double lambda = mutationRate * numGenes;  // ≈ 1.0
    std::poisson_distribution<int> poissonDist(lambda);
    std::uniform_int_distribution<int> posDist(0, numGenes - 1);
    int k = poissonDist(rng);
    for (int m = 0; m < k; ++m) {
        flipBit(ind, posDist(rng));
    }
}

// ─── Fitness (Phase 1+2: equal-weight column-sum with bit scanning) ───────────

// Exact fitness via pre-centered returns: O(T × n) where n = selected count.
// Mathematically: w^T Σ_sub w = ||Xc_sub @ w||² / (T-1).
// For equal weights: = ||column_sum||² / (n² × (T-1)).
inline double calculateFitnessExact(const BitIndividual& individual, int numGenes,
                              const Eigen::MatrixXd& centeredReturns,
                              const Eigen::VectorXd& expectedReturns,
                              int T,
                              int minETFs, int maxETFs,
                              double riskFreeRate, double minReturn) {
    int n = popcount(individual);
    if (n < minETFs || n > maxETFs) return -1e4;

    double sumER = 0.0;
    Eigen::VectorXd portSeries = Eigen::VectorXd::Zero(T);

    // Bit-scanning: iterate only over set bits
    for (size_t w = 0; w < individual.size(); ++w) {
        uint64_t bits = individual[w];
        while (bits) {
            int bit = __builtin_ctzll(bits);
            int idx = static_cast<int>(w) * BITS_PER_WORD + bit;
            if (idx < numGenes) {
                sumER += expectedReturns(idx);
                portSeries += centeredReturns.col(idx);
            }
            bits &= bits - 1;  // clear lowest set bit
        }
    }

    double portfolioReturn = sumER / n;
    if (minReturn >= 0 && portfolioReturn < minReturn) return -1e4;

    // ||portSeries||² / (n² × (T-1)) = equal-weight portfolio variance
    double portfolioVar = portSeries.squaredNorm() /
        (static_cast<double>(n) * n * (T - 1));
    if (portfolioVar <= 0) return -1e4;
    double portfolioRisk = std::sqrt(portfolioVar) * std::sqrt(252.0);

    return (portfolioReturn - riskFreeRate) / portfolioRisk;
}

// Phase 5: Approximate fitness via truncated SVD projection.
// SV is k × numCols where k << T. Column accumulation in k-dim instead of T-dim.
inline double calculateFitnessSVD(const BitIndividual& individual, int numGenes,
                            const Eigen::MatrixXd& svMatrix,
                            const Eigen::VectorXd& expectedReturns,
                            int T,
                            int minETFs, int maxETFs,
                            double riskFreeRate, double minReturn) {
    int n = popcount(individual);
    if (n < minETFs || n > maxETFs) return -1e4;

    int k = static_cast<int>(svMatrix.rows());
    double sumER = 0.0;
    Eigen::VectorXd projSum = Eigen::VectorXd::Zero(k);

    for (size_t w = 0; w < individual.size(); ++w) {
        uint64_t bits = individual[w];
        while (bits) {
            int bit = __builtin_ctzll(bits);
            int idx = static_cast<int>(w) * BITS_PER_WORD + bit;
            if (idx < numGenes) {
                sumER += expectedReturns(idx);
                projSum += svMatrix.col(idx);
            }
            bits &= bits - 1;
        }
    }

    double portfolioReturn = sumER / n;
    if (minReturn >= 0 && portfolioReturn < minReturn) return -1e4;

    // ||projSum||² / (n² × (T-1)) ≈ equal-weight portfolio variance
    double portfolioVar = projSum.squaredNorm() /
        (static_cast<double>(n) * n * (T - 1));
    if (portfolioVar <= 0) return -1e4;
    double portfolioRisk = std::sqrt(portfolioVar) * std::sqrt(252.0);

    return (portfolioReturn - riskFreeRate) / portfolioRisk;
}

// ─── Migration (bitwise) ──────────────────────────────────────────────────────

struct MigrationBuffer {
    std::vector<std::vector<BitIndividual>> buffers; // [island][individual]
    std::vector<std::mutex> locks;
    int num_islands;

    MigrationBuffer(int n) : num_islands(n), buffers(n), locks(n) {}

    void deposit(int island_id, const std::vector<BitIndividual>& individuals) {
        std::lock_guard<std::mutex> lock(locks[island_id]);
        buffers[island_id] = individuals;
    }

    std::vector<BitIndividual> withdraw(int source_island) {
        std::lock_guard<std::mutex> lock(locks[source_island]);
        return buffers[source_island];
    }
};

// ─── Hall of Fame: top-N unique solutions per island ───────────────────────────

struct HallOfFameEntry {
    double fitness;
    BitIndividual individual;
};

struct HallOfFame {
    int capacity;
    std::vector<HallOfFameEntry> entries;

    explicit HallOfFame(int cap = 20) : capacity(cap) { entries.reserve(cap + 1); }

    void tryInsert(double fitness, const BitIndividual& ind) {
        if (fitness <= -1e3) return;  // skip infeasible
        // Check if we have room or this beats the worst
        if (static_cast<int>(entries.size()) >= capacity
            && fitness <= entries.back().fitness) return;
        // Dedup: reject exact duplicates
        for (const auto& e : entries) {
            if (e.individual == ind) return;
        }
        // Insert and maintain sorted order (descending by fitness)
        entries.push_back({fitness, ind});
        // Insertion sort from the back
        for (int i = static_cast<int>(entries.size()) - 1; i > 0; --i) {
            if (entries[i].fitness > entries[i - 1].fitness)
                std::swap(entries[i], entries[i - 1]);
            else
                break;
        }
        // Trim to capacity
        if (static_cast<int>(entries.size()) > capacity)
            entries.pop_back();
    }
};

// ─── Island result ─────────────────────────────────────────────────────────────

struct IslandResult {
    double bestFitness = -std::numeric_limits<double>::infinity();
    BitIndividual bestIndividual;
    long long evaluations = 0;  // actual fitness evaluations performed
    HallOfFame hallOfFame;
};

// ─── Helper: extract tickers from a BitIndividual ──────────────────────────────

inline std::vector<std::string> extractTickers(const BitIndividual& individual,
                                         int numGenes,
                                         const std::vector<std::string>& tickers) {
    std::vector<std::string> selected;
    for (size_t w = 0; w < individual.size(); ++w) {
        uint64_t bits = individual[w];
        while (bits) {
            int bit = __builtin_ctzll(bits);
            int idx = static_cast<int>(w) * BITS_PER_WORD + bit;
            if (idx < numGenes) {
                selected.push_back(tickers[idx]);
            }
            bits &= bits - 1;
        }
    }
    return selected;
}
