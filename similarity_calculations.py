import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from pearson import pearson_correlation  
from sklearn.preprocessing import MultiLabelBinarizer
import joblib

# Load the processed data
games = pd.read_csv("dataset/preprocessed_games.csv")
# Load the MultiLabelBinarizer objects

mlb_mechanics = joblib.load('mlb_mechanics.pkl')
mlb_domains = joblib.load('mlb_domains.pkl')
# Select features we care about correlation (mechanics and domains only)
mechanics_columns = list(mlb_mechanics.classes_)
domains_columns = list(mlb_domains.classes_)
features_to_correlate = mechanics_columns + domains_columns
features_to_correlate = mechanics_columns + domains_columns

favorite_game_titles = ['Dominion']
# Ensure all favorite games are in the dataset
for title in favorite_game_titles:
    if title not in games['Name'].values:
        raise ValueError(f"The game '{title}' is not found in the dataset.")
print("Recommendations for game:", favorite_game_titles)

favorite_indices = [games[games['Name'] == title].index[0] for title in favorite_game_titles]
# Get the feature vector for the favorite game
favorite_vectors = games.loc[favorite_indices, features_to_correlate].values

# PEARSON CORRELATION
# Calculate Pearson correlation between the favorite game vector and all games
correlations = np.array([pearson_correlation(favorite_vectors[i], games[features_to_correlate].values) for i in range(len(favorite_vectors))])
average_correlations = correlations.mean(axis=0)
games['pearson_correlation'] = average_correlations
# COSINE SIMILARITY
cos_similarities = cosine_similarity(favorite_vectors, games[features_to_correlate].values)
average_similarities = cos_similarities.mean(axis=0)
games['cosine_similarity'] = average_similarities

# Normalize the rating average to the range [0, 1]
scaler = MinMaxScaler()
games['Rating Average Normalized'] = scaler.fit_transform(games[['Rating Average']])

# Combine Pearson correlation and normalized rating average to favor games with higher ratings in recommendations
games['combined_score_cosine'] = 0.5 * games['cosine_similarity'] + 0.5 * games['Rating Average Normalized']
games['combined_score_pearson'] = 0.5 * games['pearson_correlation'] + 0.5 * games['Rating Average Normalized']
# Remove favorite games from recommendations
recommendations_cosine = games[~games['Name'].isin(favorite_game_titles)].sort_values(by='combined_score_cosine', ascending=False)
recommendations_pearson = games[~games['Name'].isin(favorite_game_titles)].sort_values(by='combined_score_pearson', ascending=False)

# Display the top 10 recommended games for both methods
print("Top 10 Recommended Games based on Cosine Similarity:")
print(recommendations_cosine[['Name', 'Rating Average', 'Year Published', 'combined_score_cosine']].head(10))

print("\nTop 10 Recommended Games based on Pearson Correlation:")
print(recommendations_pearson[['Name', 'Rating Average', 'Year Published', 'combined_score_pearson']].head(10))
