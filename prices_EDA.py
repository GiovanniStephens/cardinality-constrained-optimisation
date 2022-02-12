import optimisation as op
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import warnings
warnings.filterwarnings("ignore")

# import the data
df = pd.read_csv('Data/ETF_Prices.csv', index_col=0)

# Check the data types
print(f'Data types: {df.dtypes}\n')

# Print the shape of the data
print(f'Shape of the data: {df.shape}\n')

# Count number of infinity values
print(f'Number of infinity values: {df.isnull().sum()}\n')

# Count the number of columns with more than 10% missing values
null_ETFs = df.loc[:, df.isnull().sum()/len(df) > 0.1]
print(f'Number of columns with more than 10% missing values: {len(null_ETFs.columns)}\n')

# Work out the averrage length of nulls in a row
def get_longest_nulls_run(x):
    longest_run = 0
    count = 0
    for i in range(1, len(x)):
        if pd.isna(x[i]):
            count += 1
        else:
            count = 0
        if count > longest_run:
            longest_run = count
    return longest_run

runs = [get_longest_nulls_run(df[x]) for x in null_ETFs]
print(f'Description of nulls in a row for ETFs with 10% or more missing values: {pd.DataFrame(runs).describe()}\n')

# Get the percentage of missing values in each ETF
print(f'Percentage of missing values in each ETF:\n{null_ETFs.isnull().sum()/len(null_ETFs)}\n')

# Count the number of ETFs that have more than 2% missing values
print(f'Number of ETFs with more than 2% missing values: {len(df.loc[:, df.isnull().sum()/len(df) > 0.02].columns)}\n')

# get the average length of runs for ETFs with missing values less than 10%
runs = [get_longest_nulls_run(df[x]) for x in df.loc[:, (df.isnull().sum()/len(df) < 0.1) & (df.isnull().sum()/len(df) > 0.01)]]
print(f'Description of nulls in a row for ETFs with less than 10% missing values: {pd.DataFrame(runs).describe()}\n')

# Plot a line graph of the first 5 columns
df.iloc[:, 0:5].plot()
plt.show()

# Calculate log returns
log_returns = np.log(df/df.shift(1))

# Plot a line graph of the first 5 columns
log_returns.iloc[:, 0:5].plot()

# Plot a histogram of the log return means
sns.set(style="whitegrid")
sns.histplot(log_returns.mean(),
             kde=True,
             color='orange',
             label='ETF mean returns for all ETFs')
plt.title('Histogram of mean returns')
plt.xlabel('Mean returns')
plt.ylabel('Frequency')
plt.show()

# Plot a histogram of the log returns for the first ETF
sns.set(style="whitegrid")
sns.histplot(log_returns.iloc[:, 0],
             kde=True,
             color='orange',
             label='ETF mean returns for ETF 1')
plt.title('Histogram of mean returns for ETF 1')
plt.xlabel('Mean returns')
plt.ylabel('Frequency')
plt.show()

# Show a histogram of the maximum log returns
sns.set(style="whitegrid")
sns.histplot(log_returns.max(),
             kde=True,
             color='orange',
             label='Maximum log returns')
plt.title('Histogram of maximum log returns')
plt.xlabel('Maximum log returns')
plt.ylabel('Frequency')
plt.show()

# Show correlation matrix of the log returns for the first 10 ETFs
sns.set(style="whitegrid")
sns.heatmap(log_returns.iloc[:, 0:10].corr(),
            annot=True,
            cmap='RdYlGn',
            linewidths=0.5)
plt.title('Correlation matrix of log returns for the first 10 ETFs')
plt.show()

# Show the correlation matrix of the log returns of the first 10 ETFs on the last 252 days
sns.set(style="whitegrid")
sns.heatmap(log_returns.iloc[-252:, 0:10].corr(),
            annot=True,
            cmap='RdYlGn',
            linewidths=0.5)
plt.title('Correlation matrix of log returns for the first 10 ETFs on the last 252 days')
plt.show()
