#pragma once

#include <Eigen/Dense>
#include <string>
#include <vector>

// Binary format: uint32 num_rows, uint32 num_cols,
//   num_cols null-terminated ticker strings,
//   num_rows * num_cols float64 values (row-major).
// Returns the matrix directly as log returns (no further processing needed).
// Phase 4: Uses Eigen::Map for bulk row-major → col-major transpose.
Eigen::MatrixXd readBinaryData(const std::string& filename,
                                std::vector<std::string>& tickers);

Eigen::MatrixXd readETFData(const std::string& filename, std::vector<std::string>& tickers);

Eigen::MatrixXd filterETFsWithMissingData(const Eigen::MatrixXd& data,
                                           std::vector<std::string>& tickers,
                                           double missingThreshold = 0.02);

void forwardFill(Eigen::MatrixXd& matrix);

void backwardFill(Eigen::MatrixXd& matrix);

Eigen::MatrixXd calculateLogReturns(const Eigen::MatrixXd& prices);

Eigen::VectorXd calculateExpectedReturn(const Eigen::MatrixXd& returns);

// Retained for SLSQP/weighted refinement on sub-covariance matrices (n ≤ 30).
// NEVER call this on the full M×M matrix — see CLAUDE.md covariance rule.
Eigen::MatrixXd calculateCovarianceMatrix(const Eigen::MatrixXd& returns);
