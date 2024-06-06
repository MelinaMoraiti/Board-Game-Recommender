import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Load the data
games = pd.read_csv("../datasets/games.csv")
mechanics = pd.read_csv("../datasets/mechanics.csv")
themes = pd.read_csv("../datasets/themes.csv")
subcategories = pd.read_csv("../datasets/subcategories.csv")
user_ratings = pd.read_csv("../datasets/user_ratings.csv")

# Extract relevant columns (categories) from the games dataframe
categories = ['Cat:Thematic', 'Cat:War', 'Cat:Strategy', 'Cat:Family', 'Cat:CGS', 'Cat:Abstract', 'Cat:Party', 'Cat:Childrens']
categories_df = games[['BGGId'] + categories]

# Merge the datasets on BGGId
game_features = categories_df.merge(themes, on='BGGId').merge(mechanics, on='BGGId').merge(subcategories, on='BGGId')

# Set BGGId as index
game_features.set_index('BGGId', inplace=True)

# Compute the cosine similarity matrix for the game features
similarity_matrix = cosine_similarity(game_features)

# Convert the similarity matrix to a DataFrame
similarity_df = pd.DataFrame(similarity_matrix, index=game_features.index, columns=game_features.index)
print(similarity_df[178900][1353])
# Use a smaller sample of user ratings for testing
sample_size = 50
user_ratings_sample = user_ratings.sample(n=sample_size, random_state=42)

# Create the user-game matrix from the sampled user ratings
user_game_matrix = user_ratings_sample.pivot(index='Username', columns='BGGId', values='Rating').fillna(0)
# Ensure all games are included in the user-game matrix
all_game_ids = games['BGGId'].unique()
user_game_matrix = user_game_matrix.reindex(columns=all_game_ids, fill_value=0.0)

# Function to predict ratings based on item similarity
def predict_ratings(user_game_matrix, similarity_df):
    # Initialize the predictions DataFrame with float type
    predictions = pd.DataFrame(index=user_game_matrix.index, columns=user_game_matrix.columns, dtype=float).fillna(0.0)
    
    for user in user_game_matrix.index:
        user_ratings = user_game_matrix.loc[user]
        for item in user_game_matrix.columns:
            if user_ratings[item] == 0:  # Predict rating only for items the user hasn't rated
                similar_items = similarity_df[item]
                rated_items = user_ratings[user_ratings > 0].index
                if len(rated_items) > 0:
                    numerator = (similar_items[rated_items] * user_ratings[rated_items]).sum()
                    denominator = similar_items[rated_items].sum()
                    predicted_rating = numerator / denominator if denominator != 0 else user_ratings.mean()
                else:
                    predicted_rating = user_ratings.mean()  # Fallback to user's average rating if no rated items
                predictions.at[user, item] = predicted_rating
            else:
                predictions.at[user, item] = user_ratings[item]
                
    return predictions

# Function to get top N recommendations for a user
def get_top_n_recommendations(predicted_ratings, user_game_matrix, user, n=10):
    user_predictions = predicted_ratings.loc[user].sort_values(ascending=False)
    already_rated_games = user_game_matrix.loc[user][user_game_matrix.loc[user] > 0].index
    recommendations = [item for item in user_predictions.index if item not in already_rated_games]
    return recommendations[:n]
"""
# Function to get user input ratings for three games
def get_user_input_ratings():
    print("Enter your ratings for three games of your choice (1-10, 0 if not rated):")
    user_input = {}
    for _ in range(3):
        game_name = input("Enter the name of the game: ")
        rating = float(input(f"Enter your rating for {game_name}: "))
        if 0 <= rating <= 10:
            if game_name in name_to_bggid:
                user_input[name_to_bggid[game_name]] = rating
            else:
                print("Game name not found. Please try again.")
                return None
        else:
            print("Rating should be between 0 and 10. Try again.")
            return None
    return user_input
"""
name_to_bggid = dict(zip(games["Name"], games["BGGId"]))
# Helper function to convert game names to BGGIds
def convert_names_to_bggids(user_input_ratings):
    return {name_to_bggid[game_name]: rating for game_name, rating in user_input_ratings.items()}

# Function to recommend games based on user input ratings
def recommend_based_on_input(user_input_ratings, n=10):
    user_input_to_bggids = convert_names_to_bggids(user_input_ratings)
    if not user_input_to_bggids:
        print("Invalid game names provided.")
        return []
    
    # Convert user input to DataFrame
    user_input_df = pd.DataFrame(user_input_to_bggids, index=["new_user"], columns=user_game_matrix.columns).fillna(0.0)
    # Set index of user input DataFrame to "new_user"
    user_input_df.index.name = "Username"

    # Concatenate user input DataFrame with the original user-game matrix DataFrame
    updated_user_game_matrix = pd.concat([user_game_matrix, user_input_df])
    # Predict ratings for the updated user-game matrix
    updated_predicted_ratings = predict_ratings(updated_user_game_matrix, similarity_df)
    # Get top recommendations for the new user
    top_recommendations = get_top_n_recommendations(updated_predicted_ratings, updated_user_game_matrix, "new_user", n)
    return top_recommendations

# Example usage
user_input_ratings = {"Tsuro": 8.4, "Azul": 9, "Sagrada": 9.7}

top_recommendations = recommend_based_on_input(user_input_ratings, n=10)

# Print the top recommendations with game names
print(f"Top 10 recommendations for the new user:")
for bggid in top_recommendations:
    print(f"{bggid}: {games.loc[games['BGGId'] == bggid, 'Name'].values[0]}")

