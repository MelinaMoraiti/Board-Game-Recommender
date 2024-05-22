import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer, MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from pearson import pearson_correlation
games = pd.read_excel("dataset/BGG_Data_Set.xlsx")

# Replace zeroes in 'years' with nan
games["Year Published"].replace(0, np.nan, inplace=True)
median_year = games['Year Published'].median()
# Replace zeroes in 'years' with median
games['Year Published'].fillna(median_year, inplace=True)
# Replace empty strings in 'domains' with 'Unknown' split 'mechanics' and 'domains' into lists.
# Correct way to fill missing values
games['Domains'] = games['Domains'].fillna('Unknown')
games['Mechanics'] = games['Mechanics'].fillna('Unknown')

# Split 'Mechanics' and 'Domains' columns into lists
games['Domains'] = games['Domains'].apply(lambda x: [domain.strip().lower() for domain in x.split(',')])
games['Mechanics'] = games['Mechanics'].apply(lambda x: [mechanic.strip().lower() for mechanic in x.split(',')])

# Use MultiLabelBinarizer to create binary features
mlb_mechanics = MultiLabelBinarizer()
mlb_domains = MultiLabelBinarizer()
mechanics_encoded = pd.DataFrame(mlb_mechanics.fit_transform(games['Mechanics']), columns=mlb_mechanics.classes_, index=games.index)
domains_encoded = pd.DataFrame(mlb_domains.fit_transform(games['Domains']), columns=mlb_domains.classes_, index=games.index)

# Concatenate encoded features with the original DataFrame
games = pd.concat([games, mechanics_encoded, domains_encoded], axis=1)

# Select features we care about correlation
features_to_correlate = list(mechanics_encoded.columns) + list(domains_encoded.columns) 

favorite_game_titles = ['Forbidden Island']  
# Ensure all favorite games are in the dataset
for title in favorite_game_titles:
    if title not in games['Name'].values:
        raise ValueError(f"The game '{title}' is not found in the dataset.")
print("Recommendations for game:", favorite_game_titles)

favorite_indices = [games[games['Name'] == title].index[0] for title in favorite_game_titles]
# Get the feature vector for the favorite game
favorite_vectors = games.loc[favorite_indices, features_to_correlate].values

# Calculate Pearson correlation between the favorite game vector and all games
correlations = np.array([pearson_correlation(favorite_vectors[i], games[features_to_correlate].values) for i in range(len(favorite_vectors))])
average_correlations = correlations.mean(axis=0)
games['pearson_correlation'] = average_correlations

cos_similarities = cosine_similarity(favorite_vectors,games[features_to_correlate].values)
average_similarities = cos_similarities.mean(axis=0)
games['cosine_similarity'] = average_similarities

# Normalize the rating average to the range [0, 1]
scaler = MinMaxScaler()
games['Rating Average Normalized'] = scaler.fit_transform(games[['Rating Average']])

# Combine Pearson correlation and normalized rating average to favour games with higher ratings in recommendations
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



