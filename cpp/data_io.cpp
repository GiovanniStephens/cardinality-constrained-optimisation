#include "data_io.h"
#include <fstream>
#include <iostream>
#include <cmath>
#include <limits>
#include "csv.hpp"

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

Eigen::MatrixXd filterETFsWithMissingData(const Eigen::MatrixXd& data,
                                           std::vector<std::string>& tickers,
                                           double missingThreshold) {
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
