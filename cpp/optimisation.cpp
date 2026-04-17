#include "data_io.h"
#include "ga_types.h"
#include "monte_carlo.h"

#include <Eigen/Dense>
#include <Eigen/SVD>
#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <chrono>
#include <thread>
#include <mutex>
#include <numeric>
#include <cassert>
#include <barrier>
#include <memory>

#ifdef __APPLE__
#include <dispatch/dispatch.h>
#endif

#ifdef HAS_METAL
#include "metal_fitness.h"
#endif

Config parse_args(int argc, char* argv[]) {
    Config cfg;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: optimisation [options]\n"
                << "  --mode MODE            Algorithm: ga or mc (default: ga)\n"
                << "  --data PATH            Price CSV (default: data/ETF_Prices.csv)\n"
                << "  --pop-size N           GA: population per island (default: 1000)\n"
                << "  --generations N        GA: generations per island (default: 50)\n"
                << "  --min-etfs N           Minimum ETFs per portfolio (default: 3)\n"
                << "  --max-etfs N           Maximum ETFs per portfolio (default: 15)\n"
                << "  --risk-free-rate R     Risk-free rate (default: 0.0)\n"
                << "  --seed N               Random seed, -1 for random (default: -1)\n"
                << "  --num-islands N        Threads (GA islands or MC workers) (default: auto)\n"
                << "  --time-budget S        Time budget in seconds, -1 for none (default: -1)\n"
                << "  --min-return R         Minimum portfolio return, -1 for none (default: -1)\n"
                << "  --num-elites N         GA: elites per island (default: 10)\n"
                << "  --migration-interval N GA: generations between migrations (default: 10)\n"
                << "  --migration-rate R     GA: fraction of population to migrate (default: 0.1)\n"
                << "  --top-k N              Top K solutions to output (default: 100)\n"
                << "  --missing-threshold R  Max fraction of NaN rows per column (default: 0.02)\n"
                << "  --mc-log-interval N    MC: log every N trials per thread (default: 5000)\n"
                << "  --binary               Read binary format instead of CSV\n"
                << "  --svd                  Use truncated SVD for approximate fitness\n"
                << "  --svd-components N     Number of SVD components (default: 200)\n"
                << "  --gpu                  Use Metal GPU for fitness evaluation\n"
                << "  --mutation-initial R   Adaptive mutation: initial rate (default: 0.02)\n"
                << "  --mutation-final R     Adaptive mutation: final rate (default: 0.005)\n"
                << "  --stagnation-restart N Reinitialise island after N stagnant gens (default: 200, 0=off)\n";
            std::exit(0);
        }
        // Boolean flags (no value)
        if (arg == "--binary") { cfg.binary_input = true; continue; }
        if (arg == "--svd") { cfg.use_svd = true; continue; }
        if (arg == "--gpu") { cfg.use_gpu = true; continue; }
        if (i + 1 >= argc) break;
        std::string val = argv[++i];
        if (arg == "--mode") cfg.mode = val;
        else if (arg == "--data") cfg.data_path = val;
        else if (arg == "--pop-size") cfg.pop_size = std::stoi(val);
        else if (arg == "--generations") cfg.num_generations = std::stoi(val);
        else if (arg == "--min-etfs") cfg.min_etfs = std::stoi(val);
        else if (arg == "--max-etfs") cfg.max_etfs = std::stoi(val);
        else if (arg == "--risk-free-rate") cfg.risk_free_rate = std::stod(val);
        else if (arg == "--seed") cfg.seed = std::stoi(val);
        else if (arg == "--num-islands") cfg.num_islands = std::stoi(val);
        else if (arg == "--time-budget") cfg.time_budget = std::stod(val);
        else if (arg == "--min-return") cfg.min_return = std::stod(val);
        else if (arg == "--num-elites") cfg.num_elites = std::stoi(val);
        else if (arg == "--migration-interval") cfg.migration_interval = std::stoi(val);
        else if (arg == "--migration-rate") cfg.migration_rate = std::stod(val);
        else if (arg == "--top-k") cfg.top_k = std::stoi(val);
        else if (arg == "--missing-threshold") cfg.missing_threshold = std::stod(val);
        else if (arg == "--mc-log-interval") cfg.mc_log_interval = std::stoi(val);
        else if (arg == "--svd-components") cfg.svd_components = std::stoi(val);
        else if (arg == "--mutation-initial") cfg.mutation_initial = std::stod(val);
        else if (arg == "--mutation-final") cfg.mutation_final = std::stod(val);
        else if (arg == "--stagnation-restart") cfg.stagnation_restart = std::stoi(val);
    }
    if (cfg.num_islands < 0)
        cfg.num_islands = static_cast<int>(std::thread::hardware_concurrency());
    return cfg;
}

// ─── Island GA (bitwise) ──────────────────────────────────────────────────────

void run_island(int id, const Config& cfg,
                const Eigen::MatrixXd& centeredReturns,
                const Eigen::VectorXd& expectedReturns,
                const Eigen::MatrixXd* svMatrix,  // nullptr if not using SVD
                int numGenes, int T,
                MigrationBuffer& migration,
                std::chrono::steady_clock::time_point deadline,
                bool hasDeadline,
                std::mutex& outputMutex,
                IslandResult& result,
                void* gpuEvaluator = nullptr) {

    // Per-island seeded RNG
    unsigned int islandSeed;
    if (cfg.seed >= 0) {
        islandSeed = static_cast<unsigned int>(cfg.seed + id);
    } else {
        std::random_device rd;
        islandSeed = rd();
    }
    std::mt19937 rng(islandSeed);

    auto population = initializePopulation(cfg.pop_size, numGenes, cfg.min_etfs, cfg.max_etfs, rng);
    Eigen::VectorXd fitness = Eigen::VectorXd::Zero(cfg.pop_size);
    BitIndividual bestIndividual(numWords(numGenes), 0);
    double bestFitness = -std::numeric_limits<double>::infinity();

    // Adaptive mutation: linear decay from initial to final rate
    double mutationInitial = std::max(cfg.mutation_initial, 1.0 / numGenes);
    double mutationFinal = std::max(cfg.mutation_final, 0.5 / numGenes);
    double mutationRate = mutationInitial;

    // Stagnation tracking for restart
    int stagnationCounter = 0;
    int totalRestarts = 0;

    int numElites = std::max(0, cfg.num_elites);
    // With 0 elites, use the full population as parents (no selection pressure)
    int numParents = (numElites == 0) ? cfg.pop_size : std::max(2, numElites);
    int migrationCount = std::max(1, static_cast<int>(cfg.pop_size * cfg.migration_rate));
    long long evaluations = 0;

    // Hall of fame: track top-K unique solutions across all generations
    HallOfFame hallOfFame(cfg.top_k);

    // Pre-allocate buffers reused every generation (avoid per-generation heap allocs)
    std::vector<int> parentIdx(cfg.pop_size);
    std::vector<int> repairBuf;
    repairBuf.reserve(numGenes);
    std::vector<BitIndividual> newPop;
    newPop.reserve(cfg.pop_size);
    Eigen::VectorXd newFitness = Eigen::VectorXd::Zero(cfg.pop_size);

#ifdef HAS_METAL
    // Pre-allocate GPU flat bit vector buffer (reused every generation)
    int gpuWpi = 0;
    std::vector<uint32_t> flatBits;
    if (gpuEvaluator != nullptr) {
        gpuWpi = static_cast<MetalFitnessEvaluator*>(gpuEvaluator)->wordsPerInd32();
        flatBits.resize(cfg.pop_size * gpuWpi, 0);
    }
    std::vector<double> gpuResults;
    if (gpuEvaluator != nullptr) {
        gpuResults.resize(cfg.pop_size);
    }
#endif

    // First generation: evaluate all individuals
    bool firstGeneration = true;

    for (int generation = 0; generation < cfg.num_generations; ++generation) {
        // Check time budget
        if (hasDeadline && std::chrono::steady_clock::now() >= deadline) break;

        // Evaluate fitness — skip elites after the first generation
        int evalStart = (firstGeneration || numElites == 0) ? 0 : numElites;
        int evalCount = cfg.pop_size - evalStart;

#ifdef HAS_METAL
        if (gpuEvaluator != nullptr) {
            auto* gpu = static_cast<MetalFitnessEvaluator*>(gpuEvaluator);
            int wpi = gpu->wordsPerInd32();

            // Flatten population[evalStart..evalStart+evalCount) into uint32_t buffer.
            // Each uint64 word becomes 2 uint32 words (low half first).
            // wpi == ind.size()*2 always, so every word pair is written.
            std::fill(flatBits.begin(), flatBits.begin() + evalCount * wpi, 0);
            for (int i = 0; i < evalCount; ++i) {
                const auto& ind = population[evalStart + i];
                uint32_t* dst = flatBits.data() + i * wpi;
                for (size_t w64 = 0; w64 < ind.size(); ++w64) {
                    uint64_t val = ind[w64];
                    dst[w64 * 2]     = static_cast<uint32_t>(val);
                    dst[w64 * 2 + 1] = static_cast<uint32_t>(val >> 32);
                }
            }

            // GPU batch evaluation
            gpu->evaluateBatch(flatBits.data(), evalCount, gpuResults.data());

            for (int i = 0; i < evalCount; ++i) {
                fitness(evalStart + i) = gpuResults[i];
            }
        } else
#endif
        {
            // CPU fallback (existing code)
            for (int i = evalStart; i < cfg.pop_size; ++i) {
                double f;
                if (svMatrix != nullptr) {
                    f = calculateFitnessSVD(population[i], numGenes, *svMatrix,
                                             expectedReturns, T,
                                             cfg.min_etfs, cfg.max_etfs,
                                             cfg.risk_free_rate, cfg.min_return);
                } else {
                    f = calculateFitnessExact(population[i], numGenes, centeredReturns,
                                               expectedReturns, T,
                                               cfg.min_etfs, cfg.max_etfs,
                                               cfg.risk_free_rate, cfg.min_return);
                }
                fitness(i) = f;
            }
        }
        firstGeneration = false;

        // Adaptive mutation: linear decay from initial to final rate
        {
            double progress = static_cast<double>(generation)
                            / std::max(1, cfg.num_generations - 1);
            mutationRate = mutationInitial * (1.0 - progress)
                         + mutationFinal * progress;
        }

        // Track stagnation for restart (bestFitness/bestIndividual already
        // updated in the eval loop above)
        double prevBest = bestFitness;
        for (int i = evalStart; i < cfg.pop_size; ++i) {
            if (fitness(i) > bestFitness) {
                bestFitness = fitness(i);
                bestIndividual = population[i];
            }
        }
        evaluations += evalCount;  // count batch

        // Insert generation's best into hall of fame
        hallOfFame.tryInsert(bestFitness, bestIndividual);

        // Insert top elites into hall of fame for diversity
        {
            std::vector<int> topIdx(cfg.pop_size);
            std::iota(topIdx.begin(), topIdx.end(), 0);
            int nInsert = std::min(5, cfg.pop_size);
            std::partial_sort(topIdx.begin(), topIdx.begin() + nInsert, topIdx.end(),
                              [&fitness](int a, int b) { return fitness(a) > fitness(b); });
            for (int i = 0; i < nInsert; ++i) {
                hallOfFame.tryInsert(fitness(topIdx[i]), population[topIdx[i]]);
            }
        }

        if (bestFitness > prevBest) {
            stagnationCounter = 0;
        } else {
            stagnationCounter++;
        }

        // Stagnation restart: reinitialise island (keep best individual)
        if (cfg.stagnation_restart > 0
            && stagnationCounter >= cfg.stagnation_restart) {
            totalRestarts++;
            {
                std::lock_guard<std::mutex> lock(outputMutex);
                std::cerr << "Island " << id
                          << ": Restart #" << totalRestarts
                          << " at generation " << generation
                          << " (stagnant " << stagnationCounter << " gens)"
                          << ", best = " << bestFitness
                          << std::endl;
            }
            // Reinitialise population but inject the best-ever individual
            population = initializePopulation(cfg.pop_size, numGenes,
                                               cfg.min_etfs, cfg.max_etfs, rng);
            population[0] = bestIndividual;
            fitness.setZero();
            fitness(0) = bestFitness;
            firstGeneration = false;  // elite slot 0 is valid
            stagnationCounter = 0;
            // Boost mutation temporarily after restart
            mutationRate = mutationInitial;
            continue;  // skip selection/crossover this generation
        }

        // Log convergence to stderr (every 50 generations to reduce I/O + lock overhead)
        if (generation % 50 == 0) {
            std::lock_guard<std::mutex> lock(outputMutex);
            std::cerr << "Island " << id
                      << ": Generation " << generation
                      << ": Best fitness = " << bestFitness
                      << " (mut=" << std::fixed << std::setprecision(4)
                      << mutationRate << ", restarts=" << totalRestarts << ")"
                      << std::defaultfloat
                      << std::endl;
        }

        // Migration (ring topology: read from island (id-1+N)%N)
        if (cfg.migration_interval > 0 && generation > 0
            && generation % cfg.migration_interval == 0) {
            // Partition: top migrationCount in front
            std::iota(parentIdx.begin(), parentIdx.end(), 0);
            auto cmp = [&fitness](int a, int b) { return fitness(a) > fitness(b); };
            std::nth_element(parentIdx.begin(),
                             parentIdx.begin() + migrationCount,
                             parentIdx.end(), cmp);
            // We also need the worst for replacement — partition from the end
            std::nth_element(parentIdx.begin() + migrationCount,
                             parentIdx.end() - migrationCount,
                             parentIdx.end(), cmp);

            // Export top individuals
            std::vector<BitIndividual> emigrants;
            for (int i = 0; i < migrationCount && i < cfg.pop_size; ++i)
                emigrants.push_back(population[parentIdx[i]]);
            migration.deposit(id, emigrants);

            // Import from source island — replace worst individuals
            int source = (id - 1 + cfg.num_islands) % cfg.num_islands;
            auto immigrants = migration.withdraw(source);
            if (!immigrants.empty()) {
                for (int i = 0; i < static_cast<int>(immigrants.size())
                                 && i < cfg.pop_size; ++i) {
                    int worstIdx = parentIdx[cfg.pop_size - 1 - i];
                    population[worstIdx] = immigrants[i];
                    fitness(worstIdx) = -std::numeric_limits<double>::infinity();
                }
            }
        }

        // Selection: partition so top numParents are in parentIdx[0..numParents-1]
        std::iota(parentIdx.begin(), parentIdx.end(), 0);
        std::nth_element(parentIdx.begin(),
                         parentIdx.begin() + numParents,
                         parentIdx.end(),
                         [&fitness](int a, int b) { return fitness(a) > fitness(b); });
        // Sort only the top numParents for deterministic elite ordering
        std::sort(parentIdx.begin(), parentIdx.begin() + numParents,
                  [&fitness](int a, int b) { return fitness(a) > fitness(b); });

        // Build new population, reusing pre-allocated vector
        newPop.clear();
        // Elitism: preserve top individuals, carry forward their fitness
        for (int i = 0; i < numElites && i < cfg.pop_size; ++i)
            newPop.push_back(population[parentIdx[i]]);

        // Crossover + mutation + repair
        int offspringCount = std::max(0, cfg.pop_size - numElites);
#ifdef __APPLE__
        {
            // Pre-generate one seed per offspring from master RNG (serial, fast)
            std::vector<uint64_t> offspringSeeds(offspringCount);
            for (auto& s : offspringSeeds) s = rng();
            // Pre-size newPop so dispatch workers can write by index
            newPop.resize(numElites + offspringCount);
            // Capture pointers to avoid block deep-copy of vectors
            const auto* popPtr = &population;
            const int* pidxPtr = parentIdx.data();
            auto* outPtr = &newPop;
            const uint64_t* seedPtr = offspringSeeds.data();
            dispatch_apply(static_cast<size_t>(offspringCount),
                dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0),
                ^(size_t i) {
                    SplitMix64 localRng(seedPtr[i]);
                    int p1 = localRng() % numParents;
                    int p2 = localRng() % numParents;
                    BitIndividual child = crossoverOne(
                        (*popPtr)[pidxPtr[p1]], (*popPtr)[pidxPtr[p2]], localRng);
                    mutateOne(child, mutationRate, numGenes, localRng);
                    std::vector<int> repBuf;
                    repairCardinality(child, numGenes, cfg.min_etfs, cfg.max_etfs,
                                      localRng, repBuf);
                    (*outPtr)[numElites + i] = std::move(child);
                });
        }
#else
        for (int i = 0; i < offspringCount; ++i) {
            int p1 = rng() % numParents;
            int p2 = rng() % numParents;
            BitIndividual child = crossoverOne(
                population[parentIdx[p1]], population[parentIdx[p2]], rng);
            mutateOne(child, mutationRate, numGenes, rng);
            newPop.push_back(std::move(child));
        }
        for (int i = numElites; i < static_cast<int>(newPop.size()); ++i)
            repairCardinality(newPop[i], numGenes, cfg.min_etfs, cfg.max_etfs, rng, repairBuf);
#endif

        // Carry forward elite fitness values into swap buffer, then swap
        newFitness.setZero();
        for (int i = 0; i < numElites && i < cfg.pop_size; ++i)
            newFitness(i) = fitness(parentIdx[i]);
        fitness.swap(newFitness);

        population = std::move(newPop);
    }

    // If SVD was used, re-evaluate best individual with exact method
    if (svMatrix != nullptr) {
        bestFitness = calculateFitnessExact(bestIndividual, numGenes,
                                             centeredReturns, expectedReturns, T,
                                             cfg.min_etfs, cfg.max_etfs,
                                             cfg.risk_free_rate, cfg.min_return);
    }

    result.bestFitness = bestFitness;
    result.bestIndividual = bestIndividual;
    result.evaluations = evaluations;
    result.hallOfFame = std::move(hallOfFame);
}

// ─── GPU-coordinated Island GA ────────────────────────────────────────────────
// All islands synchronize at barriers for a single batched GPU dispatch per
// generation, eliminating command-queue serialization and enabling CPU/GPU
// overlap.

#ifdef HAS_METAL
void run_island_gpu_coordinated(
        int id, const Config& cfg,
        const Eigen::MatrixXd& centeredReturns,
        const Eigen::VectorXd& expectedReturns,
        int numGenes, int T,
        MigrationBuffer& migration,
        std::chrono::steady_clock::time_point deadline,
        bool hasDeadline,
        std::mutex& outputMutex,
        IslandResult& result,
        // Shared GPU coordination state
        std::barrier<>& preGpuBarrier,
        std::barrier<>& postGpuBarrier,
        std::vector<uint32_t>& allFlatBits,  // shared flat buffer
        std::vector<double>& allFitness,      // shared fitness buffer
        int wpi,                               // words per individual (uint32)
        std::atomic<bool>& timeExpired,
        MetalFitnessEvaluator* gpu) {

    // Per-island seeded RNG
    unsigned int islandSeed;
    if (cfg.seed >= 0) {
        islandSeed = static_cast<unsigned int>(cfg.seed + id);
    } else {
        std::random_device rd;
        islandSeed = rd();
    }
    std::mt19937 rng(islandSeed);

    auto population = initializePopulation(cfg.pop_size, numGenes, cfg.min_etfs, cfg.max_etfs, rng);
    Eigen::VectorXd fitness = Eigen::VectorXd::Zero(cfg.pop_size);
    BitIndividual bestIndividual(numWords(numGenes), 0);
    double bestFitness = -std::numeric_limits<double>::infinity();

    double mutationInitial = std::max(cfg.mutation_initial, 1.0 / numGenes);
    double mutationFinal = std::max(cfg.mutation_final, 0.5 / numGenes);
    double mutationRate = mutationInitial;

    int stagnationCounter = 0;
    int totalRestarts = 0;

    int numElites = std::max(0, cfg.num_elites);
    int numParents = (numElites == 0) ? cfg.pop_size : std::max(2, numElites);
    int migrationCount = std::max(1, static_cast<int>(cfg.pop_size * cfg.migration_rate));
    long long evaluations = 0;

    HallOfFame hallOfFame(cfg.top_k);

    std::vector<int> parentIdx(cfg.pop_size);
    std::vector<int> repairBuf;
    repairBuf.reserve(numGenes);
    std::vector<BitIndividual> newPop;
    newPop.reserve(cfg.pop_size);
    Eigen::VectorXd newFitness = Eigen::VectorXd::Zero(cfg.pop_size);

    int sliceOffset = id * cfg.pop_size;  // offset into shared buffers

    for (int generation = 0; generation < cfg.num_generations; ++generation) {
        if (timeExpired.load(std::memory_order_relaxed)) break;

        // ── Phase 1: Flatten ENTIRE population into shared GPU buffer ──
        // Always evaluate all individuals (including elites) to keep the
        // shared buffer layout simple: island i occupies [i*pop .. (i+1)*pop).
        uint32_t* mySlice = allFlatBits.data() + sliceOffset * wpi;
        for (int i = 0; i < cfg.pop_size; ++i) {
            const auto& ind = population[i];
            uint32_t* dst = mySlice + i * wpi;
            for (size_t w64 = 0; w64 < ind.size(); ++w64) {
                uint64_t val = ind[w64];
                dst[w64 * 2]     = static_cast<uint32_t>(val);
                dst[w64 * 2 + 1] = static_cast<uint32_t>(val >> 32);
            }
        }

        // ── Barrier: all islands done writing ──
        preGpuBarrier.arrive_and_wait();

        // ── Phase 2: Island 0 dispatches ONE GPU eval for all islands ──
        if (id == 0) {
            if (hasDeadline && std::chrono::steady_clock::now() >= deadline) {
                timeExpired.store(true, std::memory_order_relaxed);
            }
            if (!timeExpired.load(std::memory_order_relaxed)) {
                int totalPop = cfg.num_islands * cfg.pop_size;
                gpu->evaluateBatch(allFlatBits.data(), totalPop,
                                   allFitness.data());
            }
        }

        // ── Barrier: GPU results ready ──
        postGpuBarrier.arrive_and_wait();

        if (timeExpired.load(std::memory_order_relaxed)) break;

        // ── Phase 3: Each island reads its fitness slice ──
        double* fitSlice = allFitness.data() + sliceOffset;
        for (int i = 0; i < cfg.pop_size; ++i) {
            fitness(i) = fitSlice[i];
        }
        evaluations += cfg.pop_size;

        // ── Phase 4: CPU-only GA work (parallel across islands) ──

        // Adaptive mutation
        {
            double progress = static_cast<double>(generation)
                            / std::max(1, cfg.num_generations - 1);
            mutationRate = mutationInitial * (1.0 - progress)
                         + mutationFinal * progress;
        }

        // Best tracking + stagnation
        double prevBest = bestFitness;
        for (int i = 0; i < cfg.pop_size; ++i) {
            if (fitness(i) > bestFitness) {
                bestFitness = fitness(i);
                bestIndividual = population[i];
            }
        }

        // Insert generation's best into hall of fame
        hallOfFame.tryInsert(bestFitness, bestIndividual);

        // Insert top elites into hall of fame for diversity
        {
            std::vector<int> topIdx(cfg.pop_size);
            std::iota(topIdx.begin(), topIdx.end(), 0);
            int nInsert = std::min(5, cfg.pop_size);
            std::partial_sort(topIdx.begin(), topIdx.begin() + nInsert, topIdx.end(),
                              [&fitness](int a, int b) { return fitness(a) > fitness(b); });
            for (int i = 0; i < nInsert; ++i) {
                hallOfFame.tryInsert(fitness(topIdx[i]), population[topIdx[i]]);
            }
        }

        if (bestFitness > prevBest) {
            stagnationCounter = 0;
        } else {
            stagnationCounter++;
        }

        // Stagnation restart
        if (cfg.stagnation_restart > 0
            && stagnationCounter >= cfg.stagnation_restart) {
            totalRestarts++;
            {
                std::lock_guard<std::mutex> lock(outputMutex);
                std::cerr << "Island " << id
                          << ": Restart #" << totalRestarts
                          << " at generation " << generation
                          << " (stagnant " << stagnationCounter << " gens)"
                          << ", best = " << bestFitness
                          << std::endl;
            }
            population = initializePopulation(cfg.pop_size, numGenes,
                                               cfg.min_etfs, cfg.max_etfs, rng);
            population[0] = bestIndividual;
            fitness.setZero();
            fitness(0) = bestFitness;
            stagnationCounter = 0;
            mutationRate = mutationInitial;
            continue;
        }

        // Log
        if (generation % 50 == 0) {
            std::lock_guard<std::mutex> lock(outputMutex);
            std::cerr << "Island " << id
                      << ": Generation " << generation
                      << ": Best fitness = " << bestFitness
                      << " (mut=" << std::fixed << std::setprecision(4)
                      << mutationRate << ", restarts=" << totalRestarts << ")"
                      << std::defaultfloat
                      << std::endl;
        }

        // Migration
        if (cfg.migration_interval > 0 && generation > 0
            && generation % cfg.migration_interval == 0) {
            std::iota(parentIdx.begin(), parentIdx.end(), 0);
            auto cmp = [&fitness](int a, int b) { return fitness(a) > fitness(b); };
            std::nth_element(parentIdx.begin(),
                             parentIdx.begin() + migrationCount,
                             parentIdx.end(), cmp);
            std::nth_element(parentIdx.begin() + migrationCount,
                             parentIdx.end() - migrationCount,
                             parentIdx.end(), cmp);

            std::vector<BitIndividual> emigrants;
            for (int i = 0; i < migrationCount && i < cfg.pop_size; ++i)
                emigrants.push_back(population[parentIdx[i]]);
            migration.deposit(id, emigrants);

            int source = (id - 1 + cfg.num_islands) % cfg.num_islands;
            auto immigrants = migration.withdraw(source);
            if (!immigrants.empty()) {
                for (int i = 0; i < static_cast<int>(immigrants.size())
                                 && i < cfg.pop_size; ++i) {
                    int worstIdx = parentIdx[cfg.pop_size - 1 - i];
                    population[worstIdx] = immigrants[i];
                    fitness(worstIdx) = -std::numeric_limits<double>::infinity();
                }
            }
        }

        // Selection
        std::iota(parentIdx.begin(), parentIdx.end(), 0);
        std::nth_element(parentIdx.begin(),
                         parentIdx.begin() + numParents,
                         parentIdx.end(),
                         [&fitness](int a, int b) { return fitness(a) > fitness(b); });
        std::sort(parentIdx.begin(), parentIdx.begin() + numParents,
                  [&fitness](int a, int b) { return fitness(a) > fitness(b); });

        // Build new population
        newPop.clear();
        for (int i = 0; i < numElites && i < cfg.pop_size; ++i)
            newPop.push_back(population[parentIdx[i]]);

        int offspringCount = std::max(0, cfg.pop_size - numElites);
        {
            std::vector<uint64_t> offspringSeeds(offspringCount);
            for (auto& s : offspringSeeds) s = rng();
            newPop.resize(numElites + offspringCount);
            const auto* popPtr = &population;
            const int* pidxPtr = parentIdx.data();
            auto* outPtr = &newPop;
            const uint64_t* seedPtr = offspringSeeds.data();
            dispatch_apply(static_cast<size_t>(offspringCount),
                dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0),
                ^(size_t i) {
                    SplitMix64 localRng(seedPtr[i]);
                    int p1 = localRng() % numParents;
                    int p2 = localRng() % numParents;
                    BitIndividual child = crossoverOne(
                        (*popPtr)[pidxPtr[p1]], (*popPtr)[pidxPtr[p2]], localRng);
                    mutateOne(child, mutationRate, numGenes, localRng);
                    std::vector<int> repBuf;
                    repairCardinality(child, numGenes, cfg.min_etfs, cfg.max_etfs,
                                      localRng, repBuf);
                    (*outPtr)[numElites + i] = std::move(child);
                });
        }

        newFitness.setZero();
        for (int i = 0; i < numElites && i < cfg.pop_size; ++i)
            newFitness(i) = fitness(parentIdx[i]);
        fitness.swap(newFitness);

        population = std::move(newPop);
    }

    result.bestFitness = bestFitness;
    result.bestIndividual = bestIndividual;
    result.evaluations = evaluations;
    result.hallOfFame = std::move(hallOfFame);
}
#endif  // HAS_METAL

// ─── JSON helpers ──────────────────────────────────────────────────────────────

std::string escapeJson(const std::string& s) {
    std::string out;
    for (char c : s) {
        if (c == '"') out += "\\\"";
        else if (c == '\\') out += "\\\\";
        else out += c;
    }
    return out;
}

// ─── Main ──────────────────────────────────────────────────────────────────────

int main(int argc, char* argv[]) {
    Config cfg = parse_args(argc, argv);

    // ── Input validation ────────────────────────────────────────────────────
    if (cfg.min_etfs > cfg.max_etfs) {
        std::cerr << "Error: min-etfs (" << cfg.min_etfs
                  << ") must be <= max-etfs (" << cfg.max_etfs << ")." << std::endl;
        return 1;
    }
    if (cfg.max_etfs > 64) {
        std::cerr << "Error: max-etfs (" << cfg.max_etfs
                  << ") must be <= 64 (Metal shader limitation)." << std::endl;
        return 1;
    }
    if (cfg.pop_size <= 0) {
        std::cerr << "Error: pop-size must be > 0." << std::endl;
        return 1;
    }
    if (cfg.num_islands <= 0) {
        std::cerr << "Error: num-islands must be > 0." << std::endl;
        return 1;
    }

    auto globalStart = std::chrono::steady_clock::now();

    // Load data
    std::vector<std::string> tickers;
    Eigen::MatrixXd logReturns;
    Eigen::VectorXd expectedReturns;
    int numETFs;

    if (cfg.binary_input) {
        std::cerr << "Loading binary data from " << cfg.data_path << "..." << std::endl;
        logReturns = readBinaryData(cfg.data_path, tickers);
        numETFs = logReturns.cols();
        expectedReturns = calculateExpectedReturn(logReturns);
    } else {
        std::cerr << "Loading CSV data from " << cfg.data_path << "..." << std::endl;
        Eigen::MatrixXd etfData = readETFData(cfg.data_path, tickers);
        Eigen::MatrixXd filteredData = filterETFsWithMissingData(etfData, tickers, cfg.missing_threshold);
        forwardFill(filteredData);
        backwardFill(filteredData);
        numETFs = filteredData.cols();
        logReturns = calculateLogReturns(filteredData);
        expectedReturns = calculateExpectedReturn(logReturns);
    }
    int T = static_cast<int>(logReturns.rows());
    std::cerr << "Loaded " << numETFs << " instruments, "
              << T << " return observations." << std::endl;

    // ── Post-load validation ────────────────────────────────────────────────
    if (numETFs <= 0) {
        std::cerr << "Error: no instruments loaded (M=0)." << std::endl;
        return 1;
    }
    if (T <= 0) {
        std::cerr << "Error: no return observations (T=0)." << std::endl;
        return 1;
    }
    if (cfg.max_etfs > numETFs) {
        std::cerr << "Error: max-etfs (" << cfg.max_etfs
                  << ") exceeds number of instruments (" << numETFs << ")." << std::endl;
        return 1;
    }

    // Phase 1: Pre-center returns once (avoids redundant centering per fitness call)
    Eigen::MatrixXd centeredReturns = logReturns.rowwise() - logReturns.colwise().mean();

    // Phase 5: Optional truncated SVD approximation
    Eigen::MatrixXd svMatrix;  // k × numCols
    Eigen::MatrixXd* svMatrixPtr = nullptr;
    if (cfg.use_svd && numETFs >= 5000) {
        int k = std::min(cfg.svd_components,
                         std::min(T, numETFs));
        std::cerr << "Computing truncated SVD (k=" << k << ")..." << std::endl;
        auto svdStart = std::chrono::steady_clock::now();
        Eigen::BDCSVD<Eigen::MatrixXd> svd(centeredReturns,
                                             Eigen::ComputeThinV);
        // SV = diag(S_k) @ V_k^T  →  k × numCols matrix
        svMatrix = svd.singularValues().head(k).asDiagonal()
                 * svd.matrixV().leftCols(k).transpose();
        auto svdElapsed = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - svdStart).count();
        std::cerr << "SVD computed in " << svdElapsed << "s ("
                  << k << " components, captures "
                  << svd.singularValues().head(k).squaredNorm() /
                     svd.singularValues().squaredNorm() * 100.0
                  << "% of variance)." << std::endl;
        svMatrixPtr = &svMatrix;
    } else if (cfg.use_svd) {
        std::cerr << "SVD skipped: only " << numETFs
                  << " instruments (threshold: 5000)." << std::endl;
    }

    // GPU fitness evaluator (Metal)
#ifdef HAS_METAL
    std::unique_ptr<MetalFitnessEvaluator> gpuEvaluator;
    if (cfg.use_gpu) {
        gpuEvaluator = std::make_unique<MetalFitnessEvaluator>(
            centeredReturns.data(), T, numETFs,
            expectedReturns.data(),
            cfg.min_etfs, cfg.max_etfs,
            cfg.risk_free_rate, cfg.min_return);
        if (!gpuEvaluator->isValid()) {
            std::cerr << "Metal: GPU init failed, falling back to CPU." << std::endl;
            gpuEvaluator.reset();
        }
    }
    void* gpuEvalPtr = gpuEvaluator.get();
#else
    void* gpuEvalPtr = nullptr;
    if (cfg.use_gpu)
        std::cerr << "Warning: --gpu ignored (not compiled with Metal support)." << std::endl;
#endif

    // Time budget starts AFTER data loading and preprocessing
    bool hasDeadline = cfg.time_budget > 0;
    auto computeStart = std::chrono::steady_clock::now();
    auto deadline = computeStart + std::chrono::milliseconds(
        static_cast<long long>(cfg.time_budget * 1000));

    std::mutex outputMutex;

    struct Solution {
        double fitness;
        std::vector<std::string> selectedTickers;
    };
    std::vector<Solution> allSolutions;
    long long totalTrials = 0;

    int numThreads = cfg.num_islands;

    if (cfg.mode == "mc") {
        // ── Monte Carlo mode ───────────────────────────────────────────────
        std::cerr << "Mode: Monte Carlo (" << numThreads << " threads)" << std::endl;
        std::vector<std::thread> threads;
        std::vector<MCResult> mcResults(numThreads);

        for (int i = 0; i < numThreads; ++i) {
            threads.emplace_back(run_mc_worker, i, std::cref(cfg),
                                 std::cref(centeredReturns),
                                 std::cref(expectedReturns),
                                 svMatrixPtr, numETFs, T,
                                 deadline, hasDeadline,
                                 std::ref(outputMutex), std::ref(mcResults[i]));
        }
        for (auto& t : threads) t.join();

        for (int i = 0; i < numThreads; ++i) {
            totalTrials += mcResults[i].trials;
            if (mcResults[i].bestFitness <= -1e3) continue;
            Solution sol;
            sol.fitness = mcResults[i].bestFitness;
            sol.selectedTickers = extractTickers(mcResults[i].bestIndividual,
                                                  numETFs, tickers);
            allSolutions.push_back(sol);
        }

    } else {
        // ── GA mode ────────────────────────────────────────────────────────
        std::cerr << "Mode: Island GA (" << numThreads << " islands)" << std::endl;
        MigrationBuffer migration(numThreads);
        std::vector<std::thread> threads;
        std::vector<IslandResult> gaResults(numThreads);

        for (int i = 0; i < numThreads; ++i) {
            threads.emplace_back(run_island, i, std::cref(cfg),
                                 std::cref(centeredReturns),
                                 std::cref(expectedReturns),
                                 svMatrixPtr, numETFs, T,
                                 std::ref(migration), deadline, hasDeadline,
                                 std::ref(outputMutex), std::ref(gaResults[i]),
                                 gpuEvalPtr);
        }
        for (auto& t : threads) t.join();

        // Merge halls of fame from all islands
        std::vector<HallOfFameEntry> mergedHof;
        for (int i = 0; i < numThreads; ++i) {
            totalTrials += gaResults[i].evaluations;
            for (auto& e : gaResults[i].hallOfFame.entries)
                mergedHof.push_back(std::move(e));
        }
        // Sort descending by fitness
        std::sort(mergedHof.begin(), mergedHof.end(),
                  [](const HallOfFameEntry& a, const HallOfFameEntry& b) {
                      return a.fitness > b.fitness;
                  });
        // Dedup across islands (exact bit-vector match)
        std::vector<HallOfFameEntry> dedupHof;
        for (auto& e : mergedHof) {
            bool dup = false;
            for (const auto& d : dedupHof) {
                if (d.individual == e.individual) { dup = true; break; }
            }
            if (!dup) dedupHof.push_back(std::move(e));
        }

#ifdef HAS_METAL
        // Re-evaluate with FP64 for precise final reporting
        if (gpuEvaluator.get()) {
            for (auto& e : dedupHof) {
                e.fitness = calculateFitnessExact(
                    e.individual, numETFs,
                    centeredReturns, expectedReturns, T,
                    cfg.min_etfs, cfg.max_etfs,
                    cfg.risk_free_rate, cfg.min_return);
            }
            // Re-sort after FP64 re-evaluation
            std::sort(dedupHof.begin(), dedupHof.end(),
                      [](const HallOfFameEntry& a, const HallOfFameEntry& b) {
                          return a.fitness > b.fitness;
                      });
        }
#endif

        for (auto& e : dedupHof) {
            if (e.fitness <= -1e3) continue;
            Solution sol;
            sol.fitness = e.fitness;
            sol.selectedTickers = extractTickers(e.individual, numETFs, tickers);
            allSolutions.push_back(std::move(sol));
        }
    }

    // gpuEvaluator cleaned up automatically by unique_ptr

    std::sort(allSolutions.begin(), allSolutions.end(),
              [](const Solution& a, const Solution& b) { return a.fitness > b.fitness; });

    int topK = std::min(cfg.top_k, static_cast<int>(allSolutions.size()));

    // Output JSON to stdout (full double precision)
    // Use computeStart (excludes data loading) for elapsed_seconds
    auto elapsed = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - computeStart).count();

    std::cout << std::setprecision(15);
    std::cout << "{" << std::endl;
    std::cout << "  \"mode\": \"" << cfg.mode << "\"," << std::endl;
    std::cout << "  \"elapsed_seconds\": " << elapsed << "," << std::endl;
    std::cout << "  \"num_threads\": " << numThreads << "," << std::endl;
    std::cout << "  \"num_instruments\": " << numETFs << "," << std::endl;
    std::cout << "  \"total_trials\": " << totalTrials << "," << std::endl;
    std::cout << "  \"best_fitness\": "
              << (allSolutions.empty() ? -1e9 : allSolutions[0].fitness) << "," << std::endl;

    // Best solution tickers
    std::cout << "  \"selected_tickers\": [";
    if (!allSolutions.empty()) {
        for (size_t i = 0; i < allSolutions[0].selectedTickers.size(); ++i) {
            if (i > 0) std::cout << ", ";
            std::cout << "\"" << escapeJson(allSolutions[0].selectedTickers[i]) << "\"";
        }
    }
    std::cout << "]," << std::endl;

    // Top-K solutions
    std::cout << "  \"top_solutions\": [" << std::endl;
    for (int k = 0; k < topK; ++k) {
        std::cout << "    {\"fitness\": " << allSolutions[k].fitness << ", \"tickers\": [";
        for (size_t i = 0; i < allSolutions[k].selectedTickers.size(); ++i) {
            if (i > 0) std::cout << ", ";
            std::cout << "\"" << escapeJson(allSolutions[k].selectedTickers[i]) << "\"";
        }
        std::cout << "]}";
        if (k + 1 < topK) std::cout << ",";
        std::cout << std::endl;
    }
    std::cout << "  ]" << std::endl;
    std::cout << "}" << std::endl;

    return 0;
}
