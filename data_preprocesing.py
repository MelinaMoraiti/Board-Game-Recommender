import pandas as pd
from math import sqrt
import numpy as np

games = pd.read_csv('dataset/games.csv')[['BGGId','Name']]
mechanics = pd.read_csv('dataset/mechanics.csv').drop(columns=['BGGId'])
themes = pd.read_csv('dataset/themes.csv').drop(columns=['BGGId'])

boardgames_df = pd.concat([games, mechanics, themes], axis=1)

boardgames_df.to_csv('dataset/boardgames_df.csv', index=False)