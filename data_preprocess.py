import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer

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

# Save the preprocessed data
games.to_csv("dataset/preprocessed_games.csv", index=False)




