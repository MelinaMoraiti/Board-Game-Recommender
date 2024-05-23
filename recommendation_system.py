import pandas as pd
from sklearn.cluster import KMeans
from elbow_method import plot_elbow_method
from silhouette_method import plot_silhouette_scores
from K_Means import kmeans
import numpy as np
from pearson import pearson_correlation

# Load the preprocessed data
boardgames = pd.read_csv("dataset/preprocessed_boardgames.csv")

# Extracting the features for clustering (excluding the 'Name' column)
# Ensure only numeric data is selected for clustering
features = boardgames.drop(columns=['Name']).select_dtypes(include=[np.number])

# Fill NaN values with 0 or use a different strategy if appropriate
features = features.fillna(0)

# Plotting the elbow method to help determine the optimal k
# plot_elbow_method(features)

# Plotting the silhouette scores to help determine the optimal k
# plot_silhouette_scores(features)

# Applying K-means clustering
# Determine the number of clusters (k) using the elbow method or other techniques
k = 5  # Example value; you may want to determine the optimal k

kmeans = KMeans(n_clusters=k, random_state=42)
features['Cluster'] = kmeans.fit_predict(features)

# Saving the clustered results
features.to_csv("dataset/clustered_boardgames.csv", index=False)

favorite_game_titles = ['Wingspan']
# Ensure all favorite games are in the dataset
for title in favorite_game_titles:
    if title not in boardgames['Name'].values:
        raise ValueError(f"The game '{title}' is not found in the dataset.")
print("Recommendations for game:", favorite_game_titles)

favorite_indices = [boardgames[boardgames['Name'] == title].index[0] for title in favorite_game_titles]

# Get the feature vector for the favorite game
favorite_vectors = features.loc[favorite_indices].values

# PEARSON CORRELATION
# Calculate Pearson correlation between the favorite game vector and all games
correlations = np.array([pearson_correlation(favorite_vectors[i], features.values) for i in range(len(favorite_vectors))])
average_correlations = correlations.mean(axis=0)
boardgames['pearson_correlation'] = average_correlations

# Remove favorite games from recommendations
recommendations_pearson = boardgames[~boardgames['Name'].isin(favorite_game_titles)].sort_values(by='pearson_correlation', ascending=False)

# Display the top 10 recommended games
print("\nTop 10 Recommended Games based on Pearson Correlation:")
print(recommendations_pearson[['Name']].head(10))
