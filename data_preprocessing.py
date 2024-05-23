import pandas as pd
from math import sqrt
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer

boardgames_df = pd.read_excel('dataset/BGG_Data_Set.xlsx')

#filling missing values
boardgames_df['Year Published'] = boardgames_df['Year Published'].replace(0, np.nan)
boardgames_df['Mechanics'] = boardgames_df['Mechanics'].fillna('Unknown')
boardgames_df['Domains'] = boardgames_df['Domains'].fillna('Unknown')

#splitting the mechanics and domains columns / list of strings
boardgames_df['Mechanics'] = boardgames_df['Mechanics'].apply(lambda x: x.split(', ') if x != 'Unknown' else [])
boardgames_df['Domains'] = boardgames_df['Domains'].apply(lambda x: x.split(', ') if x != 'Unknown' else [])

#creating new dataframe with only the needeed columns
boardgames_df = boardgames_df[['Name', 'Mechanics', 'Domains']]

#multiLabelBinarizer to transform lists into one-hot encoded vectors
mlb_mechanics = MultiLabelBinarizer()
mlb_domains = MultiLabelBinarizer()

mechanics_encoded = pd.DataFrame(mlb_mechanics.fit_transform(boardgames_df['Mechanics']))
domains_encoded = pd.DataFrame(mlb_domains.fit_transform(boardgames_df['Domains']))

boardgames = pd.concat([ boardgames_df['Name'], mechanics_encoded, domains_encoded], axis=1)

#exporting the new preprocessed dataframe to a csv file
boardgames.to_csv("dataset/preprocessed_boardgames.csv", index=False)

