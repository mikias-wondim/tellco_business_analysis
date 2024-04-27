import matplotlib.pyplot as plt
import seaborn as sns

def missing_value_heatmap(dataframe):
    """
    Create heatmap plot to visualize missing values in each of the columns in the dataframe:
    
    Parameters:
        dataframe to be visualized
    """
    plt.figure(figsize=(10, 6))
    sns.heatmap(dataframe.isnull(), cmap='viridis', cbar=False)
    plt.title('Missing Values Heatmap')
    plt.show()

def plot_xdr_sessions_histogram(xdr_sessions):
    """
    Generates and displays a histogram visualizing the distribution of the
    number of xDR sessions per user.

    Parameters:
    xdr_sessions (array-like): A 1D array containing the number of xDR sessions
        for each user.

    Returns:
    None (Displays the histogram plot)
    """
    
    plt.figure(figsize=(10, 5))
    plt.hist(xdr_sessions, bins=20, color='skyblue', edgecolor='black')
    plt.title('Number of xDR Sessions per User')
    plt.xlabel('Number of Sessions')
    plt.ylabel('Frequency')
    plt.grid(False)
    plt.show()

def visualize_distributions(user_data):
    """
    Visualizes the distribution of features for each cluster using violin plots.

    Parameters:
        user_data (pandas.DataFrame): The DataFrame containing user data with cluster labels.
    """
    # Choose features to visualize (replace with your desired features)
    features_to_visualize = ['Dur. (ms)', 'Total Traffic', 'Session Frequency']

    sns.set_style('whitegrid')  # Adjust plot style (optional)

    # Create violin plots for each feature
    for feature in features_to_visualize:
        sns.violinplot(x="Cluster", y=feature, showmeans=True, data=user_data)
        plt.title(f"Distribution of {feature} across Clusters")
        plt.show()

def plot_pie_cluster_stats(cluster_stats, feature_name, stat_to_plot):
    """
    Plots the mean values for a chosen feature across clusters as a pie chart.

    Parameters:
        cluster_stats (pandas.DataFrame): The DataFrame containing cluster-wise statistics.
        feature_name (str): The name of the feature to plot (e.g., 'Dur. (ms)', 'Total Traffic').
    """
    # Check if feature exists in DataFrame
    if feature_name not in cluster_stats.columns:
        raise ValueError(f"Feature '{feature_name}' not found in cluster_stats DataFrame.")

    # Extract mean values for the chosen feature
    stat_values = cluster_stats[feature_name][stat_to_plot].tolist()

    # Define pie chart colors (adjust colors as needed)
    colors = ['blue', 'green', 'red']  # Assuming three clusters

    # Create a pie chart with cluster labels and colors
    plt.figure(figsize=(6, 6))  # Adjust figure size for pie charts
    plt.pie(stat_values, labels=cluster_stats['Cluster'], autopct="%1.1f%%", colors=colors, startangle=140)
    plt.title(f'{feature_name} per Cluster - {stat_to_plot} ')
    plt.axis('equal')  # Equal aspect ratio for a circular pie chart
    plt.show()

def plot_bar_cluster_stats(cluster_stats, feature_name, stat_to_plot):
    """
    Plots the mean values for a chosen feature across clusters.

    Parameters:
        cluster_stats (pandas.DataFrame): The DataFrame containing cluster-wise statistics.
        feature_name (str): The name of the feature to plot (e.g., 'Dur. (ms)', 'Total Traffic').
    """

    # Check if feature exists in DataFrame
    if feature_name not in cluster_stats.columns:
        raise ValueError(f"Feature '{feature_name}' not found in cluster_stats DataFrame.")

    # Extract cluster labels and stats values for the chosen feature
    cluster_labels = cluster_stats['Cluster'].tolist()
    stat_values = cluster_stats[feature_name][stat_to_plot].tolist()

    # Define a color list for each cluster 
    colors = ['blue', 'green', 'red'] 

    # Create a bar graph with cluster labels and colors
    plt.figure(figsize=(8, 6))
    plt.bar(cluster_labels, stat_values, color=colors, label=cluster_labels)
    plt.xlabel('Cluster')
    plt.ylabel(f'{feature_name} - {stat_to_plot} ')  
    plt.title(f'{feature_name} per Cluster - {stat_to_plot} ')
    plt.legend()
    plt.xticks(rotation=45) 
    plt.tight_layout()
    plt.show()

def plot_scatter_cluster(result, features_to_visualize):
    """
    Plot a scatter plot for the cluster in the features

    Parameters: 
        result (pandas.DataFrame): The DataFrame containing cluster-wise results.
        feature_name (str): The name of the feature to plot (e.g., 'Dur. (ms)', 'Total Traffic').
    """
    # Select data for plotting (assuming 'Cluster' is the cluster label column)
    data_to_visualize = result[features_to_visualize]
    cluster_labels = result['Cluster']

    # Define marker shapes and transparency for each cluster
    markers = ['o', 's', '^']  
    transparency = [0.4, 0.6, 0.8]  

    # Create a scatter plot with markers and transparency
    plt.figure(figsize=(8, 6))
    for i, cluster in enumerate(cluster_labels.unique()):
        data_subset = data_to_visualize.loc[cluster_labels == cluster]
        plt.scatter(data_subset[features_to_visualize[0]], data_subset[features_to_visualize[1]], 
                label=f'Cluster {cluster}', marker=markers[i], alpha=transparency[i])

    # Add labels and title
    plt.xlabel(features_to_visualize[0])
    plt.ylabel(features_to_visualize[1])
    plt.title('K-Means Clusters (Markers)')
    plt.legend()
    plt.show()

def plot_clusters(df, cluster_centers):
    """
    Plot the clusters in a scatter plot.

    Args:
        df (pandas.DataFrame): DataFrame with cluster labels.

    Returns:
        None
    """
    # Extract feature columns
    features = ['Dur. (ms)', 'Total Traffic', 'Session Frequency']

    # Plot the data points colored by cluster labels
    plt.figure(figsize=(10, 6))
    for i in range(len(cluster_centers)):
        cluster_data = df[df['Cluster'] == i]
        plt.scatter(cluster_data[features[0]], cluster_data[features[1]], label=f'Cluster {i}')

   
    plt.xlabel('Total Session Duration (ms)')
    plt.ylabel('Total Traffic (Bytes)')
    plt.title('Cluster Analysis')
    plt.legend()
    plt.grid(True)
    plt.show()


def cluster_plot_2d(cluster_data, x_feature, y_feature):
    """
    Create a scatter plot with hue based on cluster data
    """
    # Check if the features are found on the cluster column
    if x_feature not in cluster_data.columns:
        print(f'{x_feature} is not a feature on the cluster data columns')
        return
    if y_feature not in cluster_data.columns:
        print(f'{y_feature} is not a feature on the cluster data columns')
        return

    sns.scatterplot(
        x=x_feature,
        y=y_feature,
        hue="Cluster",
        style="Cluster", 
        data=cluster_data
    )

    # Add title and labels
    plt.title('Distribution of User Experience Metrics by Cluster (2D)')
    plt.xlabel('Average TCP Retransmission Total')
    plt.ylabel('Average RTT Total')
    plt.show()

def plot_cluster_dispersion_subplots(cluster_data, metric_cols, nrows=3, ncols=4, figsize=(18, 9)):
    """
    This function plots user experience metric dispersions (assuming means) 
    across clusters in subplots.

    Args:
        cluster_data (pandas.DataFrame): The DataFrame containing cluster dispersion data.
        metric_cols(list): The user experience metric column names 
        nrows (int, optional): Number of rows in the subplot grid (defaults to 3).
        ncols (int, optional): Number of columns in the subplot grid (defaults to 4).
        figsize (tuple, optional): Size of the figure containing the subplots (defaults to (18, 9)).
    """

    # Ensure enough subplots for all metrics (optional)
    num_subplots = nrows * ncols
    if len(metric_cols) > num_subplots:
        print(f"Warning: More metrics ({len(metric_cols)}) than available subplots ({num_subplots}). Some metrics might not be plotted.")

    # Create a figure and subplots grid
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)

    # Counter to track subplot position
    plot_count = 0

    # Loop through each user experience metric
    for metric in metric_cols:
        # Extract metric name and assumed dispersion property (assuming column format)
        metric_name, dispersion_property = metric 

        # Select data for the current metric (assuming means)
        metric_data = cluster_data[metric]

        # Check if we've reached the subplot limit or the end of metrics
        if plot_count >= num_subplots or len(metric_cols) == plot_count:
            break

        # Calculate row and column index within the subplot grid
        row_index = plot_count // ncols
        col_index = plot_count % ncols

        # Create bar plot on the current subplot
        metric_data.plot(kind='bar', ax=axes[row_index, col_index], x=metric_data.index, title=f'{dispersion_property.upper()}')
        axes[row_index, col_index].set_xlabel('Cluster')
        axes[row_index, col_index].set_ylabel(metric_name)  # Assuming y-axis label same as metric name

        # Increase counter for subplot position
        plot_count += 1

    # Adjust spacing within subplots (optional)
    plt.tight_layout()
    # Show all subplots at once
    plt.show()