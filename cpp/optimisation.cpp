#include <Eigen/Dense>
#include <Eigen/SVD>
#include <fstream>
#include <sstream>
#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <random>
#include <algorithm>
#include <cmath>
#include <chrono>
#include <thread>
#include <mutex>
#include <atomic>
#include <numeric>
#include "csv.hpp"

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
    int top_k = 5;
    double missing_threshold = 0.02; // fraction of rows allowed to be NaN
    int mc_log_interval = 5000;      // MC: log every N trials per thread
    bool binary_input = false;       // if true, read binary format instead of CSV
    bool use_svd = false;            // if true, use truncated SVD for fitness
    int svd_components = 200;        // number of SVD components to keep
};

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
                << "  --top-k N              Top K solutions to output (default: 5)\n"
                << "  --missing-threshold R  Max fraction of NaN rows per column (default: 0.02)\n"
                << "  --mc-log-interval N    MC: log every N trials per thread (default: 5000)\n"
                << "  --binary               Read binary format instead of CSV\n"
                << "  --svd                  Use truncated SVD for approximate fitness\n"
                << "  --svd-components N     Number of SVD components (default: 200)\n";
            std::exit(0);
        }
        // Boolean flags (no value)
        if (arg == "--binary") { cfg.binary_input = true; continue; }
        if (arg == "--svd") { cfg.use_svd = true; continue; }
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
    }
    if (cfg.num_islands < 0)
        cfg.num_islands = static_cast<int>(std::thread::hardware_concurrency());
    return cfg;
}

// ─── Data I/O ──────────────────────────────────────────────────────────────────

// Binary format: uint32 num_rows, uint32 num_cols,
//   num_cols null-terminated ticker strings,
//   num_rows * num_cols float64 values (row-major).
// Returns the matrix directly as log returns (no further processing needed).
// Phase 4: Uses Eigen::Map for bulk row-major → col-major transpose.
Eigen::MatrixXd readBinaryData(const std::string& filename,
                                std::vector<std::string>& tickers) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) throw std::runtime_error("Cannot open binary file: " + filename);

    uint32_t numRows, numCols;
    file.read(reinterpret_cast<char*>(&numRows), 4);
    file.read(reinterpret_cast<char*>(&numCols), 4);

    tickers.clear();
    tickers.reserve(numCols);
    for (uint32_t i = 0; i < numCols; ++i) {
        std::string ticker;
        char c;
        while (file.get(c) && c != '\0')
            ticker += c;
        tickers.push_back(ticker);
    }

    // Read row-major float64 data into buffer
    std::vector<double> buf(static_cast<size_t>(numRows) * numCols);
    file.read(reinterpret_cast<char*>(buf.data()),
              static_cast<std::streamsize>(numRows) * numCols * sizeof(double));

    // Map row-major buffer and convert to col-major Eigen matrix in one bulk op
    Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
        rowMajor(buf.data(), numRows, numCols);
    return Eigen::MatrixXd(rowMajor);
}

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

// Retained for SLSQP/weighted refinement on sub-covariance matrices (n ≤ 30).
// NEVER call this on the full M×M matrix — see CLAUDE.md covariance rule.
Eigen::MatrixXd calculateCovarianceMatrix(const Eigen::MatrixXd& returns) {
    int n = returns.rows();
    Eigen::MatrixXd centered = returns.rowwise() - returns.colwise().mean();
    return (centered.transpose() * centered) / (n - 1);
}

// ─── Bitwise individual representation (Phase 2) ──────────────────────────────

using BitIndividual = std::vector<uint64_t>;
static constexpr int BITS_PER_WORD = 64;

inline int numWords(int numGenes) {
    return (numGenes + BITS_PER_WORD - 1) / BITS_PER_WORD;
}

inline bool getBit(const BitIndividual& ind, int pos) {
    return (ind[pos / BITS_PER_WORD] >> (pos % BITS_PER_WORD)) & 1ULL;
}

inline void setBit(BitIndividual& ind, int pos) {
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

// ─── GA operators (bitwise) ────────────────────────────────────────────────────

std::vector<BitIndividual> initializePopulation(int size, int numGenes,
                                                  int maxNumETFs, std::mt19937& rng) {
    int nw = numWords(numGenes);
    double prob = static_cast<double>(maxNumETFs) / numGenes;
    std::bernoulli_distribution dist(prob);
    std::vector<BitIndividual> population(size, BitIndividual(nw, 0));

    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < numGenes; ++j) {
            if (dist(rng)) setBit(population[i], j);
        }
    }
    return population;
}

// Uniform crossover: for each 64-bit word, generate a random mask and blend
BitIndividual crossoverOne(const BitIndividual& p1, const BitIndividual& p2,
                            std::mt19937& rng) {
    BitIndividual child(p1.size());
    for (size_t w = 0; w < p1.size(); ++w) {
        // Generate full 64-bit random mask from two 32-bit rng calls
        uint64_t mask = static_cast<uint64_t>(rng()) |
                        (static_cast<uint64_t>(rng()) << 32);
        child[w] = (p1[w] & mask) | (p2[w] & ~mask);
    }
    return child;
}

// Phase 3: Poisson mutation — O(1) expected per individual instead of O(numGenes)
void mutateOne(BitIndividual& ind, double mutationRate, int numGenes,
               std::mt19937& rng) {
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
double calculateFitnessExact(const BitIndividual& individual, int numGenes,
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
double calculateFitnessSVD(const BitIndividual& individual, int numGenes,
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

// ─── Island result ─────────────────────────────────────────────────────────────

struct IslandResult {
    double bestFitness = -std::numeric_limits<double>::infinity();
    BitIndividual bestIndividual;
};

// ─── Helper: extract tickers from a BitIndividual ──────────────────────────────

std::vector<std::string> extractTickers(const BitIndividual& individual,
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

    auto population = initializePopulation(cfg.pop_size, numGenes, cfg.max_etfs, rng);
    Eigen::VectorXd fitness = Eigen::VectorXd::Zero(cfg.pop_size);
    BitIndividual bestIndividual(numWords(numGenes), 0);
    double bestFitness = -std::numeric_limits<double>::infinity();
    double mutationRate = 1.0 / numGenes;

    int numElites = std::max(1, cfg.num_elites);
    int numParents = std::max(2, numElites);
    int migrationCount = std::max(1, static_cast<int>(cfg.pop_size * cfg.migration_rate));

    for (int generation = 0; generation < cfg.num_generations; ++generation) {
        // Check time budget
        if (hasDeadline && std::chrono::steady_clock::now() >= deadline) break;

        // Evaluate fitness
        for (int i = 0; i < cfg.pop_size; ++i) {
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
            if (f > bestFitness) {
                bestFitness = f;
                bestIndividual = population[i];
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
            // Sort by fitness
            std::vector<int> sortedIdx(cfg.pop_size);
            std::iota(sortedIdx.begin(), sortedIdx.end(), 0);
            std::sort(sortedIdx.begin(), sortedIdx.end(),
                      [&fitness](int a, int b) { return fitness(a) > fitness(b); });

            // Export top individuals
            std::vector<BitIndividual> emigrants;
            for (int i = 0; i < migrationCount && i < cfg.pop_size; ++i)
                emigrants.push_back(population[sortedIdx[i]]);
            migration.deposit(id, emigrants);

            // Import from source island
            int source = (id - 1 + cfg.num_islands) % cfg.num_islands;
            auto immigrants = migration.withdraw(source);
            if (!immigrants.empty()) {
                for (int i = 0; i < static_cast<int>(immigrants.size())
                                 && i < cfg.pop_size; ++i) {
                    int worstIdx = sortedIdx[cfg.pop_size - 1 - i];
                    population[worstIdx] = immigrants[i];
                }
            }
        }

        // Selection: pick top parents by fitness
        std::vector<int> parentIdx(cfg.pop_size);
        std::iota(parentIdx.begin(), parentIdx.end(), 0);
        std::sort(parentIdx.begin(), parentIdx.end(),
                  [&fitness](int a, int b) { return fitness(a) > fitness(b); });

        // Elitism: preserve top individuals
        std::vector<BitIndividual> newPop;
        newPop.reserve(cfg.pop_size);
        for (int i = 0; i < numElites && i < cfg.pop_size; ++i)
            newPop.push_back(population[parentIdx[i]]);

        // Crossover: fill remaining slots
        int offspringCount = cfg.pop_size - numElites;
        for (int i = 0; i < offspringCount; ++i) {
            int p1 = rng() % numParents;
            int p2 = rng() % numParents;
            BitIndividual child = crossoverOne(
                population[parentIdx[p1]], population[parentIdx[p2]], rng);
            mutateOne(child, mutationRate, numGenes, rng);
            newPop.push_back(std::move(child));
        }

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
}

// ─── Monte Carlo worker (bitwise) ─────────────────────────────────────────────

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
                                 std::ref(outputMutex), std::ref(gaResults[i]));
        }
        for (auto& t : threads) t.join();

        for (int i = 0; i < numThreads; ++i) {
            if (gaResults[i].bestFitness <= -1e3) continue;
            Solution sol;
            sol.fitness = gaResults[i].bestFitness;
            sol.selectedTickers = extractTickers(gaResults[i].bestIndividual,
                                                  numETFs, tickers);
            allSolutions.push_back(sol);
        }
    }

    std::sort(allSolutions.begin(), allSolutions.end(),
              [](const Solution& a, const Solution& b) { return a.fitness > b.fitness; });

    int topK = std::min(cfg.top_k, static_cast<int>(allSolutions.size()));

    // Output JSON to stdout (full double precision)
    auto elapsed = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - globalStart).count();

    std::cout << std::setprecision(15);
    std::cout << "{" << std::endl;
    std::cout << "  \"mode\": \"" << cfg.mode << "\"," << std::endl;
    std::cout << "  \"elapsed_seconds\": " << elapsed << "," << std::endl;
    std::cout << "  \"num_threads\": " << numThreads << "," << std::endl;
    std::cout << "  \"num_instruments\": " << numETFs << "," << std::endl;
    if (totalTrials > 0)
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
