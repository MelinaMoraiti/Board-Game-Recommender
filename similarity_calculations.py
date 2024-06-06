import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from pearson import pearson_correlation  
import joblib
""""
# Load the processed data
games = pd.read_csv("dataset/preprocessed_games.csv")
# Load the MultiLabelBinarizer objects

mlb_mechanics = joblib.load('mlb_mechanics.pkl')
mlb_domains = joblib.load('mlb_domains.pkl')
# Select features we care about correlation (mechanics and domains only)
mechanics_columns = list(mlb_mechanics.classes_)
domains_columns = list(mlb_domains.classes_)
features_to_correlate = mechanics_columns + domains_columns
"""
# Load the data
games = pd.read_csv("datasets/games.csv")
mechanics = pd.read_csv("datasets/mechanics.csv")
themes = pd.read_csv("datasets/themes.csv")
subcategories = pd.read_csv("datasets/subcategories.csv")
user_ratings = pd.read_csv("datasets/user_ratings.csv")

# Extract relevant columns (categories) from the games dataframe
categories = ['Cat:Thematic', 'Cat:War', 'Cat:Strategy', 'Cat:Family', 'Cat:CGS', 'Cat:Abstract', 'Cat:Party', 'Cat:Childrens']
categories_df = games[['BGGId'] + categories]

# Merge the datasets on BGGId
game_features = categories_df.merge(themes, on='BGGId').merge(mechanics, on='BGGId').merge(subcategories, on='BGGId')

# Set BGGId as index
game_features.set_index('BGGId', inplace=True)

# Ensure all necessary columns are present for correlation
mechanics_columns = mechanics.columns.tolist()[1:]  # Exclude BGGId
themes_columns = themes.columns.tolist()[1:]        # Exclude BGGId
subcategories_columns = subcategories.columns.tolist()[1:]  # Exclude BGGId
categories_columns = categories
# Combine all feature columns
features_to_correlate = mechanics_columns + themes_columns + subcategories_columns + categories_columns

favorite_game_titles = ['Codenames']
# Ensure all favorite games are in the dataset
for title in favorite_game_titles:
    if title not in games['Name'].values:
        raise ValueError(f"The game '{title}' is not found in the dataset.")
print("Recommendations for game:", favorite_game_titles)

favorite_indices = [games[games['Name'] == title].index[0] for title in favorite_game_titles]
# Get the feature vector for the favorite game
favorite_vectors = game_features.loc[games.loc[favorite_indices, 'BGGId']].values


# Calculate Pearson correlation between the favorite game vector and all games
correlations = np.array([pearson_correlation(favorite_vectors[i], game_features.values) for i in range(len(favorite_vectors))])
average_correlations = correlations.mean(axis=0)
games['pearson_correlation'] = average_correlations

# COSINE SIMILARITY
cos_similarities = cosine_similarity(favorite_vectors, game_features.values)
average_similarities = cos_similarities.mean(axis=0)
games['cosine_similarity'] = average_similarities

# Normalize the rating average to the range [0, 1]
scaler = MinMaxScaler()
games['Rating Average Normalized'] = scaler.fit_transform(games[['AvgRating']])

# Combine Pearson correlation and normalized rating average to favor games with higher ratings in recommendations
games['combined_score_cosine'] = 0.5 * games['cosine_similarity'] + 0.5 * games['Rating Average Normalized']
games['combined_score_pearson'] = 0.5 * games['pearson_correlation'] + 0.5 * games['Rating Average Normalized']

# Remove favorite games from recommendations
recommendations_cosine = games[~games['Name'].isin(favorite_game_titles)].sort_values(by='combined_score_cosine', ascending=False)
recommendations_pearson = games[~games['Name'].isin(favorite_game_titles)].sort_values(by='combined_score_pearson', ascending=False)

# Display the top 10 recommended games for both methods
print("Top 10 Recommended Games based on Cosine Similarity:")
print(recommendations_cosine[['Name', 'AvgRating','combined_score_cosine']].head(10))

print("\nTop 10 Recommended Games based on Pearson Correlation:")
print(recommendations_pearson[['Name', 'AvgRating', 'combined_score_pearson']].head(10))