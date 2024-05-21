import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MultiLabelBinarizer

games = pd.read_excel("dataset/BGG_Data_Set.xlsx")
games["Year Published"].replace(0, np.nan, inplace=True)

median_year = games['Year Published'].median()
# Replace zeroes in 'years' with median
games['Year Published'].fillna(median_year, inplace=True)

# Replace empty strings in 'domains' with 'Unknown' split 'mechanics' and 'domains' into lists.
games['Domains'] = games['Domains'].apply(lambda x: ['Unknown'] if pd.isna(x) or x.strip() == '' else x.split(', '))
games['Mechanics'] = games['Mechanics'].apply(lambda x: x.split(', ') if isinstance(x, str) else [])

# Feature Extraction
# Use techniques like One-Hot Encoding or TF-IDF Vectorization for categorical features such as 'mechanics' and 'domains'.
mlb_mechanics = MultiLabelBinarizer()
mlb_domains = MultiLabelBinarizer()
mechanics_encoded = mlb_mechanics.fit_transform(games['Mechanics'])
domains_encoded = mlb_domains.fit_transform(games['Domains'])

mechanics_df = pd.DataFrame(mechanics_encoded, columns=mlb_mechanics.classes_)
domains_df = pd.DataFrame(domains_encoded, columns=mlb_domains.classes_)


games = pd.concat([games, mechanics_df, domains_df], axis=1)
print(games.tail(10))
