import pandas as pd
from math import sqrt
import numpy as np

games = pd.read_csv('dataset/games.csv')[['BGGId','Name']]
mechanics = pd.read_csv('dataset/mechanics.csv')
themes = pd.read_csv('dataset/themes.csv')

boardgames_df = games.merge(mechanics, on='BGGId').merge(themes, on='BGGId')

boardgames_df.to_csv('dataset/boardgames_df.csv', index=False)