#pragma once

#include "ga_types.h"
#include <Eigen/Dense>
#include <chrono>
#include <mutex>

struct MCResult {
    double bestFitness = -std::numeric_limits<double>::infinity();
    BitIndividual bestIndividual;
    long long trials = 0;
};

void run_mc_worker(int id, const Config& cfg,
                   const Eigen::MatrixXd& centeredReturns,
                   const Eigen::VectorXd& expectedReturns,
                   const Eigen::MatrixXd* svMatrix,
                   int numGenes, int T,
                   std::chrono::steady_clock::time_point deadline,
                   bool hasDeadline,
                   std::mutex& outputMutex,
                   MCResult& result);
