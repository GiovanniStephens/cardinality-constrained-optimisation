"""Binary data format for the C++ optimiser."""

import logging
import struct

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def write_binary_data(log_returns, path):
    """Write log returns matrix in binary format for the C++ optimiser.

    Format: uint32 num_rows, uint32 num_cols, then num_cols null-terminated
    ticker strings, then num_rows * num_cols float64 values in row-major order.

    :param log_returns: DataFrame of log returns (index=dates, columns=tickers).
    :param path: output file path.
    """
    tickers = list(log_returns.columns)
    mat = log_returns.values.astype(np.float64)
    num_rows, num_cols = mat.shape

    with open(path, 'wb') as f:
        f.write(struct.pack('<II', num_rows, num_cols))
        for ticker in tickers:
            f.write(ticker.encode('utf-8') + b'\x00')
        f.write(mat.tobytes(order='C'))


def read_binary_data(path):
    """Read binary data file written by write_binary_data.

    :param path: input file path.
    :return: (log_returns DataFrame, tickers list).
    """
    try:
        with open(path, 'rb') as f:
            num_rows, num_cols = struct.unpack('<II', f.read(8))
            tickers = []
            for _ in range(num_cols):
                chars = []
                while True:
                    c = f.read(1)
                    if c == b'\x00':
                        break
                    chars.append(c)
                tickers.append(b''.join(chars).decode('utf-8'))
            expected_bytes = num_rows * num_cols * 8
            raw = f.read(expected_bytes)
            if len(raw) != expected_bytes:
                raise ValueError(
                    f"Expected {expected_bytes} bytes of matrix data, got {len(raw)}. "
                    f"File may be truncated: {path}"
                )
            data = np.frombuffer(raw, dtype=np.float64)
            mat = data.reshape(num_rows, num_cols)
    except Exception:
        logger.error("Failed to read binary data from %s", path, exc_info=True)
        raise

    return pd.DataFrame(mat, columns=tickers), tickers
