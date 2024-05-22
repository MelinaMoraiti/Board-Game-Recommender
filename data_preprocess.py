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

favorite_game_title = 'Tsuro'  # Replace with the actual favorite game name
if favorite_game_title not in games['Name'].values:
    raise ValueError(f"The game '{favorite_game_title}' is not found in the dataset.")
favorite_index = games[games['Name'] == favorite_game_title].index[0]
# Get the feature vector for the favorite game
favorite_vector = games.loc[favorite_index, features_to_correlate].values

# Calculate Pearson correlation between the favorite game vector and all games
correlations = pearson_correlation(favorite_vector, games[features_to_correlate].values)

# Normalize the rating average to the range [0, 1]
scaler = MinMaxScaler()
games['Rating Average Normalized'] = scaler.fit_transform(games[['Rating Average']])

# Add correlations to the DataFrame
games['pearson_correlation'] = correlations

# Combine Pearson correlation and normalized rating average to favour games with higher ratings in recommendations
games['recommendation_score'] = 0.5 * games['pearson_correlation'] + 0.5 * games['Rating Average Normalized']

# Display the top 10 recommended games based on Pearson correlation
recommended_games = games.sort_values(by='recommendation_score', ascending=False).head(10)
print(recommended_games[['Name', 'Rating Average', 'Year Published', 'pearson_correlation', 'Rating Average Normalized', 'recommendation_score']])


