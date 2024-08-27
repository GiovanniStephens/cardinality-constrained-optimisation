#include <Eigen/Dense>
#include <fstream>
#include <sstream>
#include <iostream>
#include <string>
#include <vector>
#include <random>
#include <algorithm>
#include <cmath>
#include "csv.hpp"
#include <thread>
#include <mutex>

Eigen::MatrixXd readETFData(const std::string& filename, std::vector<std::string>& tickers) {
    // Prepare to read the file
    csv::CSVReader reader(filename);
    std::vector<std::vector<double>> data;
    bool firstRow = true;

    // Reading each row
    for (csv::CSVRow& row : reader) { // Parse the file row by row
        if (firstRow) {
            // Skip the first column and read tickers
            for (size_t i = 1; i < row.size(); i++) {
                tickers.push_back(row[i].get<std::string>());
                data.push_back(std::vector<double>());
            }
            firstRow = false;
        } else {
            int tickerIndex = 0;
            for (size_t i = 1; i < row.size(); i++) {
                double value = row[i].is_null() ? std::nan("1") : row[i].get<double>();
                data[tickerIndex++].push_back(value);
            }
        }
    }

    // Convert vector of vectors to Eigen::MatrixXd
    if (!data.empty()) {
        size_t rows = data.front().size();
        size_t cols = data.size();
        Eigen::MatrixXd mat(rows, cols);
        for (size_t i = 0; i < cols; ++i) {
            for (size_t j = 0; j < rows; ++j) {
                mat(j, i) = data[i][j];
            }
        }
        return mat;
    }
    throw std::runtime_error("Failed to parse CSV data or data is empty.");
}

Eigen::MatrixXd filterETFsWithMissingData(const Eigen::MatrixXd& data, double missingThreshold = 0.02) {
    std::vector<int> validColumns;
    for (int i = 0; i < data.cols(); ++i) {
        int countNaNs = (data.col(i).array().isNaN()).count();
        double fractionMissing = static_cast<double>(countNaNs) / static_cast<double>(data.rows());
        if (fractionMissing < missingThreshold) {
            validColumns.push_back(i);
        }
    }

    if (validColumns.empty()) {
        std::cerr << "No valid columns found. All data may be above the missing threshold." << std::endl;
        return Eigen::MatrixXd();  // Return an empty matrix if no valid columns
    }

    // Construct a new matrix with only the valid columns
    Eigen::MatrixXd filteredData(data.rows(), validColumns.size());
    for (size_t i = 0; i < validColumns.size(); ++i) {
        filteredData.col(i) = data.col(validColumns[i]);
    }
    return filteredData;
}

void forwardFill(Eigen::MatrixXd& matrix) {
    for (int col = 0; col < matrix.cols(); ++col) {
        double lastValid = std::numeric_limits<double>::quiet_NaN(); // Start with NaN

        for (int row = 0; row < matrix.rows(); ++row) {
            if (std::isnan(matrix(row, col))) { // If current entry is NaN
                if (!std::isnan(lastValid)) { // And there is a valid number to carry forward
                    matrix(row, col) = lastValid;
                }
            } else {
                lastValid = matrix(row, col); // Update last valid entry
            }
        }
    }
}

void backwardFill(Eigen::MatrixXd& matrix) {
    for (int col = 0; col < matrix.cols(); ++col) {
        double lastValid = std::numeric_limits<double>::quiet_NaN(); // Start with NaN

        for (int row = matrix.rows() - 1; row >= 0; --row) {
            if (std::isnan(matrix(row, col))) { // If current entry is NaN
                if (!std::isnan(lastValid)) { // And there is a valid number to carry forward
                    matrix(row, col) = lastValid;
                }
            } else {
                lastValid = matrix(row, col); // Update last valid entry
            }
        }
    }
}

void fillWithZeros(Eigen::MatrixXd& matrix) {
    for (int col = 0; col < matrix.cols(); ++col) {
        for (int row = 0; row < matrix.rows(); ++row) {
            if (std::isnan(matrix(row, col))) {
                matrix(row, col) = 0.0;
            }
        }
    }
}

Eigen::MatrixXd calculateLogReturns(const Eigen::MatrixXd& prices) {
    if (prices.rows() < 2) {
        return Eigen::MatrixXd(0, prices.cols());
    }
    Eigen::MatrixXd logPrices = prices.array().log();
    Eigen::MatrixXd logReturns = logPrices.bottomRows(prices.rows() - 1) - logPrices.topRows(prices.rows() - 1);
    return logReturns;
}

Eigen::VectorXd calculateExpectedReturn(const Eigen::MatrixXd& returns) {
    Eigen::VectorXd meanReturns(returns.cols());
    for (int i = 0; i < returns.cols(); ++i) {
        double sum = 0;
        int validCount = 0;
        for (int j = 0; j < returns.rows(); ++j) {
            // Check if the value is not NaN or infinite or -inf
            if (!std::isnan(returns(j, i))) {
                sum += returns(j, i);
                validCount++;
            }
        }
        if (validCount > 0) {
            double mean = sum / validCount;
            if (std::abs(mean) < std::numeric_limits<double>::epsilon()) {  // Check if mean is extremely small
                meanReturns(i) = 0.0;  // Set mean return to 0 if it's insignificantly small
            } else {
                meanReturns(i) = mean * 252;  // Annualize the mean return
            }
        } else {
            meanReturns(i) = std::numeric_limits<double>::quiet_NaN();  // Set to NaN if all are NaN
        }
    }
    return meanReturns;
}

double calculatePortfolioReturn(const Eigen::MatrixXd& expectedReturns, const Eigen::VectorXd& weights) {
    double portfolioReturn = (expectedReturns.transpose() * weights).value();
    return portfolioReturn;
}

double calculatePortfolioRisk(const Eigen::MatrixXd& covarianceMatrix, const Eigen::VectorXd& weights) {
    double portfolioRisk = std::sqrt((weights.transpose() * covarianceMatrix * weights).value()) * std::sqrt(252);
    return portfolioRisk;
}

Eigen::MatrixXd calculateCovarianceMatrix(const Eigen::MatrixXd& returns) {
    int n = returns.rows();
    Eigen::MatrixXd centered = returns.rowwise() - returns.colwise().mean();  // Subtract the mean
    Eigen::MatrixXd covMatrix = (centered.transpose() * centered) / (n - 1);
    return covMatrix;
}

double calculateSharpeRatio(double return_, double risk, double riskFreeRate = 0.01) {
    return (return_ - riskFreeRate) / risk;
}

Eigen::MatrixXd selectSubset(const Eigen::MatrixXd& data, const std::vector<int>& indices) {
    Eigen::MatrixXd subset(data.rows(), indices.size());
    for (size_t i = 0; i < indices.size(); ++i) {
        subset.col(i) = data.col(indices[i]);
    }
    return subset;
}

Eigen::MatrixXi initializePopulation(int size, int numETFs, int maxNumETFs) {
    if (maxNumETFs > numETFs) throw std::runtime_error("maxNumETFs cannot be greater than numETFs");
    if (maxNumETFs <= 0) throw std::runtime_error("maxNumETFs must be a positive integer");

    std::random_device rd;
    std::mt19937 gen(rd());
    std::bernoulli_distribution dist(static_cast<double>(maxNumETFs) / numETFs);

    Eigen::MatrixXi population(size, numETFs);
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < numETFs; ++j) {
            population(i, j) = dist(gen) ? 1 : 0;
        }
    }
    return population;
}

Eigen::MatrixXi selectParents(const Eigen::MatrixXi& population, const Eigen::VectorXd& fitness, int numParents) {
    std::vector<int> indices(population.rows());
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(), [&fitness](int a, int b) { return fitness(a) > fitness(b); });

    Eigen::MatrixXi parents(numParents, population.cols());
    for (int i = 0; i < numParents; ++i) {
        parents.row(i) = population.row(indices[i]);
    }
    return parents;
}

Eigen::MatrixXi crossover(const Eigen::MatrixXi& parents, int offspringSize) {
    std::random_device rd;
    std::mt19937 gen(rd());
    Eigen::MatrixXi offspring(offspringSize, parents.cols());
    for (int i = 0; i < offspringSize; ++i) {
        int parent1 = gen() % parents.rows();
        int parent2 = gen() % parents.rows();
        for (int j = 0; j < parents.cols(); ++j) {
            offspring(i, j) = gen() % 2 ? parents(parent1, j) : parents(parent2, j);
        }
    }
    return offspring;
}

void mutate(Eigen::MatrixXi& offspring, double mutationRate) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::bernoulli_distribution dist(mutationRate);

    for (int i = 0; i < offspring.rows(); ++i) {
        for (int j = 0; j < offspring.cols(); ++j) {
            if (dist(gen)) {
                offspring(i, j) = 1 - offspring(i, j);
            }
        }
    }
}

Eigen::MatrixXi elitism(const Eigen::MatrixXi& population, const Eigen::VectorXd& fitness, int numElites) {
    std::vector<int> indices(population.rows());
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(), [&fitness](int a, int b) { return fitness(a) > fitness(b); });

    Eigen::MatrixXi elites(numElites, population.cols());
    for (int i = 0; i < numElites; ++i) {
        elites.row(i) = population.row(indices[i]);
    }
    return elites;
}

double calculateFitness(const Eigen::RowVectorXi& individual, const Eigen::MatrixXd& logReturns,
                        const Eigen::VectorXd& expectedReturns, double riskFreeRate = 0.045) {
    std::vector<int> selectedIndices;
    for (int i = 0; i < individual.size(); ++i) {
        if (individual(i) == 1) {
            selectedIndices.push_back(i);
        }
    }
    int numSelectedETFs = selectedIndices.size();
    if (numSelectedETFs < 3 || numSelectedETFs > 10) {
        return -1e4; // Penalize solutions outside the valid range
    }
    Eigen::MatrixXd selectedLogReturns(logReturns.rows(), numSelectedETFs);
    Eigen::VectorXd selectedExpectedReturns(numSelectedETFs);
    for (int i = 0; i < numSelectedETFs; ++i) {
        selectedLogReturns.col(i) = logReturns.col(selectedIndices[i]);
        selectedExpectedReturns(i) = expectedReturns(selectedIndices[i]);
    }
    Eigen::VectorXd weights = Eigen::VectorXd::Constant(numSelectedETFs, 1.0 / numSelectedETFs); 
    Eigen::MatrixXd covarianceMatrix = calculateCovarianceMatrix(selectedLogReturns);
    double portfolioReturn = calculatePortfolioReturn(selectedExpectedReturns, weights);
    double portfolioVariance = calculatePortfolioRisk(covarianceMatrix, weights);
    double sharpeRatio = calculateSharpeRatio(portfolioReturn, portfolioVariance, riskFreeRate);
    return sharpeRatio;
}


void run_island(int id, const Eigen::MatrixXd& logReturns, const Eigen::VectorXd& expectedReturns, int populationSize, int numGenerations, double mutationRate, int maxNumETFs, int numETFs) {
    Eigen::MatrixXi population = initializePopulation(populationSize, numETFs, maxNumETFs);
    Eigen::VectorXd fitness = Eigen::VectorXd::Zero(populationSize);

    for (int generation = 0; generation < numGenerations; ++generation) {
        // Evaluate fitness
        for (int i = 0; i < population.rows(); ++i) {
            fitness(i) = calculateFitness(population.row(i), logReturns, expectedReturns, 0.0);
        }

        // Selection, Crossover, and Mutation
        Eigen::MatrixXi parents = selectParents(population, fitness, populationSize / 100);
        Eigen::MatrixXi offspring = crossover(parents, populationSize - parents.rows());
        mutate(offspring, mutationRate);

        // Elitism
        int numElites = populationSize / 100;
        Eigen::MatrixXi elites = elitism(population, fitness, numElites);
        population.topRows(numElites) = elites;
        population.middleRows(numElites, parents.rows() - numElites) = parents.bottomRows(parents.rows() - numElites);
        population.bottomRows(offspring.rows()) = offspring;

        // Print the best fitness in the current generation
        std::cout << "Island " << id << ": Generation " << generation << ": Best fitness = " << fitness.maxCoeff() << std::endl;
        // Optionally handle migration here if implementing migration between islands
    }

    // Output the best solution found on this island
    std::cout << "Island " << id << ": Best fitness achieved = " << fitness.maxCoeff() << std::endl;
}

int main() {
    std::vector<std::string> tickers;
    Eigen::MatrixXd etfData = readETFData("Data/ETF_Prices.csv", tickers);
    Eigen::MatrixXd filteredData = filterETFsWithMissingData(etfData);
    forwardFill(filteredData);
    backwardFill(filteredData);
    int numETFs = filteredData.cols();
    Eigen::MatrixXd logReturns = calculateLogReturns(filteredData);
    Eigen::VectorXd expectedReturns = calculateExpectedReturn(logReturns);

    int numIslands = 8;
    int populationSizePerIsland = 1125;
    int numGenerations = 200;
    double mutationRate = 0.0015;
    int maxNumETFs = 10;
    std::vector<std::thread> islands;

    for (int i = 0; i < numIslands; ++i) {
        islands.emplace_back(run_island, i, std::cref(logReturns),
                             std::cref(expectedReturns), populationSizePerIsland,
                             numGenerations, mutationRate, maxNumETFs, numETFs);
    }

    // Wait for all islands to complete
    for (auto& island : islands) {
        island.join();
    }

    return 0;
}
