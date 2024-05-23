import pandas as pd
from math import sqrt
import numpy as np

boardgames_df = pd.read_excel('dataset/BGG_Data_Set.xlsx')

#filling missing values
boardgames_df['Year Published'] = boardgames_df['Year Published'].replace(0, np.nan)
boardgames_df['Mechanics'] = boardgames_df['Mechanics'].fillna('Unknown')
boardgames_df['Domains'] = boardgames_df['Domains'].fillna('Unknown')

#splitting the mechanics and domains columns / list of strings
boardgames_df['Mechanics'] = boardgames_df['Mechanics'].apply(lambda x: x.split(', ') if x != 'Unknown' else [])
boardgames_df['Domains'] = boardgames_df['Domains'].apply(lambda x: x.split(', ') if x != 'Unknown' else [])

#creating new dataframe with only the needeed columns
clean_boardgames_df = boardgames_df[['ID', 'Mechanics', 'Domains']]

#exporting the new preprocessed dataframe to a csv file
clean_boardgames_df.to_csv("dataset/preprocessed_boardgames.csv", index=False)

