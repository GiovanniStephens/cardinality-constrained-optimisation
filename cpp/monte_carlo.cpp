#include "monte_carlo.h"
#include <iostream>
#include <numeric>

void run_mc_worker(int id, const Config& cfg,
                   const Eigen::MatrixXd& centeredReturns,
                   const Eigen::VectorXd& expectedReturns,
                   const Eigen::MatrixXd* svMatrix,
                   int numGenes, int T,
                   std::chrono::steady_clock::time_point deadline,
                   bool hasDeadline,
                   std::mutex& outputMutex,
                   MCResult& result) {

    unsigned int workerSeed;
    if (cfg.seed >= 0) {
        workerSeed = static_cast<unsigned int>(cfg.seed + id);
    } else {
        std::random_device rd;
        workerSeed = rd();
    }
    std::mt19937 rng(workerSeed);
    std::uniform_int_distribution<int> numDist(cfg.min_etfs, cfg.max_etfs);

    int nw = numWords(numGenes);
    double bestFitness = -std::numeric_limits<double>::infinity();
    BitIndividual bestIndividual(nw, 0);
    long long trials = 0;

    // Pre-build index array for sampling
    std::vector<int> allIndices(numGenes);
    std::iota(allIndices.begin(), allIndices.end(), 0);

    while (true) {
        if (hasDeadline && (trials % 256 == 0)
            && std::chrono::steady_clock::now() >= deadline)
            break;

        // Generate random portfolio with exactly k instruments
        int k = numDist(rng);
        BitIndividual individual(nw, 0);
        // Fisher-Yates partial shuffle to pick k indices
        for (int i = 0; i < k; ++i) {
            std::uniform_int_distribution<int> pick(i, numGenes - 1);
            std::swap(allIndices[i], allIndices[pick(rng)]);
            setBit(individual, allIndices[i]);
        }

        double f;
        if (svMatrix != nullptr) {
            f = calculateFitnessSVD(individual, numGenes, *svMatrix,
                                     expectedReturns, T,
                                     cfg.min_etfs, cfg.max_etfs,
                                     cfg.risk_free_rate, cfg.min_return);
        } else {
            f = calculateFitnessExact(individual, numGenes, centeredReturns,
                                       expectedReturns, T,
                                       cfg.min_etfs, cfg.max_etfs,
                                       cfg.risk_free_rate, cfg.min_return);
        }
        trials++;

        if (f > bestFitness) {
            bestFitness = f;
            bestIndividual = individual;
        }

        // Log periodically
        if (trials % cfg.mc_log_interval == 0) {
            std::lock_guard<std::mutex> lock(outputMutex);
            std::cerr << "MC worker " << id
                      << ": Trial " << trials
                      << ": Best fitness = " << bestFitness
                      << std::endl;
        }
    }

    // If SVD was used, re-evaluate best with exact method
    if (svMatrix != nullptr) {
        bestFitness = calculateFitnessExact(bestIndividual, numGenes,
                                             centeredReturns, expectedReturns, T,
                                             cfg.min_etfs, cfg.max_etfs,
                                             cfg.risk_free_rate, cfg.min_return);
    }

    result.bestFitness = bestFitness;
    result.bestIndividual = bestIndividual;
    result.trials = trials;
}
