import pandas as pd

def load_and_merge_data():
    games = pd.read_csv('raw-datasets/games.csv')
    mechanics = pd.read_csv('raw-datasets/mechanics.csv')
    themes = pd.read_csv('raw-datasets/themes.csv')
    subcategories = pd.read_csv('raw-datasets/subcategories.csv')

    # Define the categories to extract from games dataframe
    categories = ['Cat:Thematic', 'Cat:War', 'Cat:Strategy', 'Cat:Family', 'Cat:CGS', 'Cat:Abstract', 'Cat:Party', 'Cat:Childrens']
    categories_df = games[['BGGId', 'Name', 'AvgRating', 'BayesAvgRating'] + categories]

    # Merge datasets on BGGId
    boardgames_df = categories_df.merge(mechanics, on='BGGId', how='left')\
                                 .merge(themes, on='BGGId', how='left')\
                                 .merge(subcategories, on='BGGId', how='left')
    # Drop the rows where at least one element is missing.
    boardgames_df.dropna()
    return boardgames_df

def save_boardgames_df(df, path='datasets/boardgames_df.csv'):
    df.to_csv(path, index=False)

if __name__ == "__main__":
    boardgames_df = load_and_merge_data()
    save_boardgames_df(boardgames_df)