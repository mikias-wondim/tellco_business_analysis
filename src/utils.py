import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans


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


def drop_columns_with_null_values(dataframe, threshold=0.3):
    """
    Drop columns with missing value more than threshold_percentage of the entries

    Parameters::
    dataframe: A pandas dataframe 
    threshold: Determine the threshold percentage for dropping columns

    Returns:
    filtered dataframe
    """
    # Calculate the percentage of missing values in each column
    missing_percentages = (dataframe.isnull().sum() / len(dataframe)) * 100

    # Identify columns that exceed the threshold
    columns_to_drop = missing_percentages[missing_percentages > threshold].index

    # Drop the identified columns from the DataFrame
    df_filtered = dataframe.drop(columns=columns_to_drop)

    return df_filtered

def get_mean(dataframe):
    """
    Returns the mean of the dataframe as a list
    """
    # Use describe() method to get summary statistics of numerical columns
    summary_stats = dataframe.describe()

    # Extract the row containing the mean statistics
    mean = summary_stats.loc['mean']

    return mean

def aggregate_metric_analyze(telecom_df):
    """
    Analyzes telecom xDR sessions data, aggregating information per user and reporting top 10 users for each engagement metric.

    Args:
        telecom_df (pandas.DataFrame): The DataFrame containing xDR session data.

    Returns:
        pandas.DataFrame: A new DataFrame with aggregated user-level statistics.
    """

    # Group data by user (MSISDN/Number)
    user_data = telecom_df.groupby('MSISDN/Number')

    # Calculate desired metrics
    user_data = user_data.agg({
        'Dur. (ms)': 'sum',  # Total session duration (ms)
        'Total DL (Bytes)': 'sum',  # Total traffic (download)
        'Total UL (Bytes)': 'sum',  # Total traffic (upload)
    })

    # Add 'Total Traffic' column
    user_data['Total Traffic'] = user_data['Total DL (Bytes)'] + user_data['Total UL (Bytes)']

    # Sort and get top 10 users for each metric
    top_10_duration = user_data.sort_values(by='Dur. (ms)', ascending=False).head(10)
    top_10_traffic = user_data.sort_values(by='Total Traffic', ascending=False).head(10)

    # Count unique entries in 'Start' per user
    user_data = telecom_df.groupby('MSISDN/Number')['Start'].nunique()  

    # Sort and get top 10 users by potentially high session frequency (days with sessions)
    top_10_sessions = user_data.sort_values(ascending=False).head(10)  

    print("Top 10 Users by Session Frequency:")
    print(top_10_sessions)

    print("\nTop 10 Users by Total Session Duration:")
    print(top_10_duration)

    print("\nTop 10 Users by Total Traffic:")
    print(top_10_traffic)

def normalize_and_cluster(df):
    """
    Normalizes engagement metrics, performs k-means clustering (k=3), and assigns cluster labels.

    Args:
        telecom_df (pandas.DataFrame): The DataFrame containing xDR session data.

    Returns:
        pandas.DataFrame: A new DataFrame with normalized metrics and cluster labels for each user.
    """

    # Group data by user (MSISDN/Number)
    grouped_data = df.groupby('MSISDN/Number').agg({
    'Dur. (ms)': 'sum',  # Total session duration (ms)
    'Total DL (Bytes)': 'sum',  # Total traffic (download)
    'Total UL (Bytes)': 'sum'  # Total traffic (upload)
    }).reset_index()

    # Get the frequency of each entry
    frequency = df.groupby('MSISDN/Number').size().reset_index(name='Session Frequency')

    # Merge the frequency into the grouped data
    result = pd.merge(grouped_data, frequency, on='MSISDN/Number')

    # Combine download and upload traffic into a single 'Total Traffic' metric
    result['Total Traffic'] = result['Total DL (Bytes)'] + result['Total UL (Bytes)']
    result.drop(columns=['Total DL (Bytes)', 'Total UL (Bytes)'], inplace=True)

    # Select features for normalization and clustering
    features = ['Dur. (ms)', 'Total Traffic', 'Session Frequency']

    # Normalize features using MinMaxScaler
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(result[features])

    # Perform k-means clustering with k=3 (3 engagement groups)
    kmeans = KMeans(n_clusters=3, random_state=42)  # Set random state for reproducibility
    kmeans.fit(scaled_data)

    # Add cluster labels to user data DataFrame
    result['Cluster'] = kmeans.labels_

    return result

def analyze_clusters(result):
    """
    Calculates minimum, maximum, average, and total values for non-normalized metrics for each cluster.

    Args:
        result (pandas.DataFrame): The DataFrame containing user data with cluster labels.

    Returns:
        pandas.DataFrame: A new DataFrame with cluster-wise statistics.
    """

    # Group data by cluster label
    cluster_data = result.groupby('Cluster')

    # Calculate statistics for non-normalized features (assuming 'Dur. (ms)', 'Total Traffic', 'Session Frequency')
    stats = cluster_data[['Dur. (ms)', 'Total Traffic', 'Session Frequency']].agg(['min', 'max', 'mean', 'sum'])

    # Add a 'Total Users' column showing the count of users in each cluster
    stats['Total Users'] = cluster_data.size()

    return stats.reset_index()

def adjust_outliers(df, cluster_centers):
    """
    Adjusts outliers in session duration and total traffic within each cluster.

    Args:
        df (pandas.DataFrame): DataFrame with normalized metrics and cluster labels.
        cluster_centers (numpy.ndarray): Cluster centers obtained from KMeans.

    Returns:
        pandas.DataFrame: DataFrame with outliers adjusted.
    """

    # Calculate thresholds for session duration and total traffic
    session_duration_threshold = 0.75 * 1e07
    total_traffic_threshold = 0.6 * 1e10

    # Iterate over clusters
    for cluster_label in df['Cluster'].unique():
        cluster_data = df[df['Cluster'] == cluster_label]

        # Identify outliers in session duration
        session_duration_outliers = cluster_data[cluster_data['Dur. (ms)'] > session_duration_threshold]
        if not session_duration_outliers.empty:
            # Replace outliers with cluster center value
            cluster_center_duration = cluster_centers[cluster_label][0]
            df.loc[session_duration_outliers.index, 'Dur. (ms)'] = cluster_center_duration

        # Identify outliers in total traffic
        total_traffic_outliers = cluster_data[cluster_data['Total Traffic'] > total_traffic_threshold]
        if not total_traffic_outliers.empty:
            # Replace outliers with cluster center value
            cluster_center_traffic = cluster_centers[cluster_label][1]
            df.loc[total_traffic_outliers.index, 'Total Traffic'] = cluster_center_traffic

    return df