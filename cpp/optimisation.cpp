#include <Eigen/Dense>
#include <fstream>
#include <sstream>
#include <iostream>
#include <string>
#include <vector>
#include <random>
#include <algorithm>
#include <cmath>
#include <chrono>
#include <thread>
#include <mutex>
#include <atomic>
#include "csv.hpp"

// ─── Configuration ─────────────────────────────────────────────────────────────

struct Config {
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
    int top_k = 5;
    double missing_threshold = 0.02; // fraction of rows allowed to be NaN
};

Config parse_args(int argc, char* argv[]) {
    Config cfg;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: optimisation [options]\n"
                << "  --data PATH            Price CSV (default: data/ETF_Prices.csv)\n"
                << "  --pop-size N           Population per island (default: 1000)\n"
                << "  --generations N        Generations per island (default: 50)\n"
                << "  --min-etfs N           Minimum ETFs per portfolio (default: 3)\n"
                << "  --max-etfs N           Maximum ETFs per portfolio (default: 15)\n"
                << "  --risk-free-rate R     Risk-free rate (default: 0.0)\n"
                << "  --seed N               Random seed, -1 for random (default: -1)\n"
                << "  --num-islands N        Number of islands, -1 for auto (default: -1)\n"
                << "  --time-budget S        Time budget in seconds, -1 for none (default: -1)\n"
                << "  --min-return R         Minimum portfolio return, -1 for none (default: -1)\n"
                << "  --num-elites N         Elites per island (default: 10)\n"
                << "  --migration-interval N Generations between migrations (default: 10)\n"
                << "  --migration-rate R     Fraction of population to migrate (default: 0.1)\n"
                << "  --top-k N              Top K solutions to output (default: 5)\n"
                << "  --missing-threshold R  Max fraction of NaN rows per column (default: 0.02)\n";
            std::exit(0);
        }
        if (i + 1 >= argc) break;
        std::string val = argv[++i];
        if (arg == "--data") cfg.data_path = val;
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
    }
    if (cfg.num_islands < 0)
        cfg.num_islands = static_cast<int>(std::thread::hardware_concurrency());
    return cfg;
}

// ─── Data I/O ──────────────────────────────────────────────────────────────────

Eigen::MatrixXd readETFData(const std::string& filename, std::vector<std::string>& tickers) {
    csv::CSVFormat format;
    format.header_row(-1);
    csv::CSVReader reader(filename, format);
    std::vector<std::vector<double>> data;
    bool firstRow = true;

    for (csv::CSVRow& row : reader) {
        if (firstRow) {
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

    if (!data.empty()) {
        size_t rows = data.front().size();
        size_t cols = data.size();
        Eigen::MatrixXd mat(rows, cols);
        for (size_t i = 0; i < cols; ++i)
            for (size_t j = 0; j < rows; ++j)
                mat(j, i) = data[i][j];
        return mat;
    }
    throw std::runtime_error("Failed to parse CSV data or data is empty.");
}

// ─── Data cleaning ─────────────────────────────────────────────────────────────

Eigen::MatrixXd filterETFsWithMissingData(const Eigen::MatrixXd& data,
                                           std::vector<std::string>& tickers,
                                           double missingThreshold = 0.02) {
    std::vector<int> validColumns;
    std::vector<std::string> filteredTickers;

    for (int i = 0; i < data.cols(); ++i) {
        int countNaNs = (data.col(i).array().isNaN()).count();
        double fractionMissing = static_cast<double>(countNaNs) / static_cast<double>(data.rows());
        if (fractionMissing < missingThreshold) {
            validColumns.push_back(i);
            filteredTickers.push_back(tickers[i]);
        }
    }

    if (validColumns.empty()) {
        std::cerr << "No valid columns found." << std::endl;
        tickers.clear();
        return Eigen::MatrixXd();
    }
    tickers = filteredTickers;
    Eigen::MatrixXd filteredData(data.rows(), validColumns.size());
    for (size_t i = 0; i < validColumns.size(); ++i)
        filteredData.col(i) = data.col(validColumns[i]);
    return filteredData;
}

void forwardFill(Eigen::MatrixXd& matrix) {
    for (int col = 0; col < matrix.cols(); ++col) {
        double lastValid = std::numeric_limits<double>::quiet_NaN();
        for (int row = 0; row < matrix.rows(); ++row) {
            if (std::isnan(matrix(row, col))) {
                if (!std::isnan(lastValid)) matrix(row, col) = lastValid;
            } else {
                lastValid = matrix(row, col);
            }
        }
    }
}

void backwardFill(Eigen::MatrixXd& matrix) {
    for (int col = 0; col < matrix.cols(); ++col) {
        double lastValid = std::numeric_limits<double>::quiet_NaN();
        for (int row = matrix.rows() - 1; row >= 0; --row) {
            if (std::isnan(matrix(row, col))) {
                if (!std::isnan(lastValid)) matrix(row, col) = lastValid;
            } else {
                lastValid = matrix(row, col);
            }
        }
    }
}

// ─── Financial calculations ────────────────────────────────────────────────────

Eigen::MatrixXd calculateLogReturns(const Eigen::MatrixXd& prices) {
    if (prices.rows() < 2) return Eigen::MatrixXd(0, prices.cols());
    Eigen::MatrixXd logPrices = prices.array().log();
    return logPrices.bottomRows(prices.rows() - 1) - logPrices.topRows(prices.rows() - 1);
}

Eigen::VectorXd calculateExpectedReturn(const Eigen::MatrixXd& returns) {
    Eigen::VectorXd meanReturns(returns.cols());
    for (int i = 0; i < returns.cols(); ++i) {
        double sum = 0;
        int validCount = 0;
        for (int j = 0; j < returns.rows(); ++j) {
            if (!std::isnan(returns(j, i))) {
                sum += returns(j, i);
                validCount++;
            }
        }
        if (validCount > 0) {
            double mean = sum / validCount;
            meanReturns(i) = (std::abs(mean) < std::numeric_limits<double>::epsilon())
                             ? 0.0 : mean * 252;
        } else {
            meanReturns(i) = std::numeric_limits<double>::quiet_NaN();
        }
    }
    return meanReturns;
}

Eigen::MatrixXd calculateCovarianceMatrix(const Eigen::MatrixXd& returns) {
    int n = returns.rows();
    Eigen::MatrixXd centered = returns.rowwise() - returns.colwise().mean();
    return (centered.transpose() * centered) / (n - 1);
}

double calculatePortfolioReturn(const Eigen::VectorXd& expectedReturns, const Eigen::VectorXd& weights) {
    return (expectedReturns.transpose() * weights).value();
}

double calculatePortfolioRisk(const Eigen::MatrixXd& covarianceMatrix, const Eigen::VectorXd& weights) {
    return std::sqrt((weights.transpose() * covarianceMatrix * weights).value()) * std::sqrt(252.0);
}

// ─── GA operators ──────────────────────────────────────────────────────────────

Eigen::MatrixXi initializePopulation(int size, int numETFs, int maxNumETFs, std::mt19937& rng) {
    std::bernoulli_distribution dist(static_cast<double>(maxNumETFs) / numETFs);
    Eigen::MatrixXi population(size, numETFs);
    for (int i = 0; i < size; ++i)
        for (int j = 0; j < numETFs; ++j)
            population(i, j) = dist(rng) ? 1 : 0;
    return population;
}

Eigen::MatrixXi selectParents(const Eigen::MatrixXi& population,
                               const Eigen::VectorXd& fitness, int numParents) {
    std::vector<int> indices(population.rows());
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(),
              [&fitness](int a, int b) { return fitness(a) > fitness(b); });
    Eigen::MatrixXi parents(numParents, population.cols());
    for (int i = 0; i < numParents; ++i)
        parents.row(i) = population.row(indices[i]);
    return parents;
}

Eigen::MatrixXi crossover(const Eigen::MatrixXi& parents, int offspringSize, std::mt19937& rng) {
    Eigen::MatrixXi offspring(offspringSize, parents.cols());
    for (int i = 0; i < offspringSize; ++i) {
        int parent1 = rng() % parents.rows();
        int parent2 = rng() % parents.rows();
        for (int j = 0; j < parents.cols(); ++j)
            offspring(i, j) = rng() % 2 ? parents(parent1, j) : parents(parent2, j);
    }
    return offspring;
}

void mutate(Eigen::MatrixXi& offspring, double mutationRate, std::mt19937& rng) {
    std::bernoulli_distribution dist(mutationRate);
    for (int i = 0; i < offspring.rows(); ++i)
        for (int j = 0; j < offspring.cols(); ++j)
            if (dist(rng))
                offspring(i, j) = 1 - offspring(i, j);
}

Eigen::MatrixXi elitism(const Eigen::MatrixXi& population,
                          const Eigen::VectorXd& fitness, int numElites) {
    std::vector<int> indices(population.rows());
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(),
              [&fitness](int a, int b) { return fitness(a) > fitness(b); });
    Eigen::MatrixXi elites(numElites, population.cols());
    for (int i = 0; i < numElites; ++i)
        elites.row(i) = population.row(indices[i]);
    return elites;
}

// ─── Fitness ───────────────────────────────────────────────────────────────────

double calculateFitness(const Eigen::RowVectorXi& individual,
                        const Eigen::MatrixXd& logReturns,
                        const Eigen::VectorXd& expectedReturns,
                        int minETFs, int maxETFs,
                        double riskFreeRate, double minReturn) {
    std::vector<int> selectedIndices;
    for (int i = 0; i < individual.size(); ++i)
        if (individual(i) == 1)
            selectedIndices.push_back(i);

    int n = static_cast<int>(selectedIndices.size());
    if (n < minETFs || n > maxETFs) return -1e4;

    Eigen::MatrixXd selectedLogReturns(logReturns.rows(), n);
    Eigen::VectorXd selectedExpectedReturns(n);
    for (int i = 0; i < n; ++i) {
        selectedLogReturns.col(i) = logReturns.col(selectedIndices[i]);
        selectedExpectedReturns(i) = expectedReturns(selectedIndices[i]);
    }

    Eigen::VectorXd weights = Eigen::VectorXd::Constant(n, 1.0 / n);
    double portfolioReturn = calculatePortfolioReturn(selectedExpectedReturns, weights);

    if (minReturn >= 0 && portfolioReturn < minReturn) return -1e4;

    Eigen::MatrixXd covMatrix = calculateCovarianceMatrix(selectedLogReturns);
    double portfolioRisk = calculatePortfolioRisk(covMatrix, weights);
    if (portfolioRisk <= 0) return -1e4;

    return (portfolioReturn - riskFreeRate) / portfolioRisk;
}

// ─── Migration ─────────────────────────────────────────────────────────────────

struct MigrationBuffer {
    std::vector<std::vector<Eigen::RowVectorXi>> buffers; // [island][individual]
    std::vector<std::mutex> locks;
    int num_islands;

    MigrationBuffer(int n) : num_islands(n), buffers(n), locks(n) {}

    void deposit(int island_id, const std::vector<Eigen::RowVectorXi>& individuals) {
        std::lock_guard<std::mutex> lock(locks[island_id]);
        buffers[island_id] = individuals;
    }

    std::vector<Eigen::RowVectorXi> withdraw(int source_island) {
        std::lock_guard<std::mutex> lock(locks[source_island]);
        return buffers[source_island];
    }
};

// ─── Island result ─────────────────────────────────────────────────────────────

struct IslandResult {
    double bestFitness = -std::numeric_limits<double>::infinity();
    Eigen::RowVectorXi bestIndividual;
};

// ─── Island GA ─────────────────────────────────────────────────────────────────

void run_island(int id, const Config& cfg,
                const Eigen::MatrixXd& logReturns,
                const Eigen::VectorXd& expectedReturns,
                int numETFs, const std::vector<std::string>& tickers,
                MigrationBuffer& migration,
                std::chrono::steady_clock::time_point deadline,
                bool hasDeadline,
                std::mutex& outputMutex,
                IslandResult& result) {

    // Per-island seeded RNG
    unsigned int islandSeed;
    if (cfg.seed >= 0) {
        islandSeed = static_cast<unsigned int>(cfg.seed + id);
    } else {
        std::random_device rd;
        islandSeed = rd();
    }
    std::mt19937 rng(islandSeed);

    Eigen::MatrixXi population = initializePopulation(cfg.pop_size, numETFs, cfg.max_etfs, rng);
    Eigen::VectorXd fitness = Eigen::VectorXd::Zero(cfg.pop_size);
    Eigen::RowVectorXi bestIndividual(numETFs);
    double bestFitness = -std::numeric_limits<double>::infinity();
    double mutationRate = 1.0 / numETFs;

    int numElites = std::max(1, cfg.num_elites);
    int numParents = std::max(2, numElites);
    int migrationCount = std::max(1, static_cast<int>(cfg.pop_size * cfg.migration_rate));

    for (int generation = 0; generation < cfg.num_generations; ++generation) {
        // Check time budget
        if (hasDeadline && std::chrono::steady_clock::now() >= deadline) break;

        // Evaluate fitness
        for (int i = 0; i < population.rows(); ++i) {
            double f = calculateFitness(population.row(i), logReturns, expectedReturns,
                                        cfg.min_etfs, cfg.max_etfs,
                                        cfg.risk_free_rate, cfg.min_return);
            fitness(i) = f;
            if (f > bestFitness) {
                bestFitness = f;
                bestIndividual = population.row(i);
            }
        }

        // Log convergence to stderr
        {
            std::lock_guard<std::mutex> lock(outputMutex);
            std::cerr << "Island " << id
                      << ": Generation " << generation
                      << ": Best fitness = " << fitness.maxCoeff()
                      << std::endl;
        }

        // Migration (ring topology: read from island (id-1+N)%N)
        if (cfg.migration_interval > 0 && generation > 0
            && generation % cfg.migration_interval == 0) {
            // Export top individuals
            std::vector<int> sortedIdx(population.rows());
            std::iota(sortedIdx.begin(), sortedIdx.end(), 0);
            std::sort(sortedIdx.begin(), sortedIdx.end(),
                      [&fitness](int a, int b) { return fitness(a) > fitness(b); });

            std::vector<Eigen::RowVectorXi> emigrants;
            for (int i = 0; i < migrationCount && i < static_cast<int>(sortedIdx.size()); ++i)
                emigrants.push_back(population.row(sortedIdx[i]));
            migration.deposit(id, emigrants);

            // Import from source island
            int source = (id - 1 + cfg.num_islands) % cfg.num_islands;
            auto immigrants = migration.withdraw(source);
            if (!immigrants.empty()) {
                // Replace worst individuals
                for (int i = 0; i < static_cast<int>(immigrants.size())
                                 && i < population.rows(); ++i) {
                    int worstIdx = sortedIdx[sortedIdx.size() - 1 - i];
                    population.row(worstIdx) = immigrants[i];
                }
            }
        }

        // Selection, Crossover, Mutation
        Eigen::MatrixXi parents = selectParents(population, fitness, numParents);
        Eigen::MatrixXi offspring = crossover(parents, cfg.pop_size - numElites, rng);
        mutate(offspring, mutationRate, rng);

        // Elitism + new population
        Eigen::MatrixXi elites = elitism(population, fitness, numElites);
        population.topRows(numElites) = elites;
        population.bottomRows(offspring.rows()) = offspring;
    }

    result.bestFitness = bestFitness;
    result.bestIndividual = bestIndividual;
}

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

    auto globalStart = std::chrono::steady_clock::now();

    // Load and preprocess data
    std::vector<std::string> tickers;
    std::cerr << "Loading data from " << cfg.data_path << "..." << std::endl;
    Eigen::MatrixXd etfData = readETFData(cfg.data_path, tickers);
    Eigen::MatrixXd filteredData = filterETFsWithMissingData(etfData, tickers, cfg.missing_threshold);
    forwardFill(filteredData);
    backwardFill(filteredData);
    int numETFs = filteredData.cols();
    Eigen::MatrixXd logReturns = calculateLogReturns(filteredData);
    Eigen::VectorXd expectedReturns = calculateExpectedReturn(logReturns);
    std::cerr << "Loaded " << numETFs << " instruments, "
              << logReturns.rows() << " return observations." << std::endl;

    // Time budget
    bool hasDeadline = cfg.time_budget > 0;
    auto deadline = globalStart + std::chrono::milliseconds(
        static_cast<long long>(cfg.time_budget * 1000));

    // Migration buffer
    MigrationBuffer migration(cfg.num_islands);
    std::mutex outputMutex;

    // Launch islands
    std::vector<std::thread> threads;
    std::vector<IslandResult> results(cfg.num_islands);

    for (int i = 0; i < cfg.num_islands; ++i) {
        threads.emplace_back(run_island, i, std::cref(cfg),
                             std::cref(logReturns), std::cref(expectedReturns),
                             numETFs, std::cref(tickers),
                             std::ref(migration), deadline, hasDeadline,
                             std::ref(outputMutex), std::ref(results[i]));
    }

    for (auto& t : threads) t.join();

    // Collect top-K unique solutions across all islands
    struct Solution {
        double fitness;
        std::vector<std::string> selectedTickers;
    };
    std::vector<Solution> allSolutions;
    for (int i = 0; i < cfg.num_islands; ++i) {
        if (results[i].bestFitness <= -1e3) continue;
        Solution sol;
        sol.fitness = results[i].bestFitness;
        for (int j = 0; j < results[i].bestIndividual.size(); ++j)
            if (results[i].bestIndividual(j) == 1)
                sol.selectedTickers.push_back(tickers[j]);
        allSolutions.push_back(sol);
    }
    std::sort(allSolutions.begin(), allSolutions.end(),
              [](const Solution& a, const Solution& b) { return a.fitness > b.fitness; });

    int topK = std::min(cfg.top_k, static_cast<int>(allSolutions.size()));

    // Output JSON to stdout
    auto elapsed = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - globalStart).count();

    std::cout << "{" << std::endl;
    std::cout << "  \"elapsed_seconds\": " << elapsed << "," << std::endl;
    std::cout << "  \"num_islands\": " << cfg.num_islands << "," << std::endl;
    std::cout << "  \"num_instruments\": " << numETFs << "," << std::endl;
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
