import pandas as pd
import numpy as np

def get_numeric_columns(dataframe):
    """
    Retrieve column names with numeric data type from the given DataFrame.

    Parameters:
    dataframe (DataFrame): The pandas DataFrame from which to retrieve column names.

    Returns:
    list: A list containing column names with numeric data type.

    Example:
    >>> df = pd.DataFrame({'A': [1, 2, 3], 'B': [4.0, 5.0, 6.0], 'C': ['a', 'b', 'c']})
    >>> get_numeric_columns(df)
    ['A', 'B']
    """
    numeric_columns = dataframe.select_dtypes(include=['int64', 'float64']).columns.tolist()
    return numeric_columns

def detect_outliers_zscore(data, threshold=3):
    """
    Detect outliers in a numerical dataset using the Z-score method.

    Parameters:
    data (array-like): The numerical data for outlier detection.
    threshold (float, optional): The threshold value for identifying outliers.
        Data points with a Z-score greater than this threshold are considered outliers.
        Default is 3.

    Returns:
    array-like: A boolean array indicating outliers. True indicates an outlier.

    Example:
    >>> data = [1, 2, 3, 100, 4, 5, 6]
    >>> detect_outliers_zscore(data)
    array([False, False, False,  True, False, False, False])
    """
    z_scores = np.abs((data - np.mean(data)) / np.std(data))
    return z_scores > threshold
