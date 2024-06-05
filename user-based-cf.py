import pandas as pd
from math import sqrt
import numpy as np

boardgames_df = pd.read_csv('dataset/boardgames_df.csv')
user_ratings_df = pd.read_csv('dataset/user_ratings.csv')

played_games = [
    {'Name':'Catan', 'rating':8.5},
    {'Name':'Wingspan', 'rating':9},
    {'Name':'Smash Up', 'rating':6.7},
    {'Name':'Ticket to Ride', 'rating':7.5},
    {'Name':'Risk', 'rating':6},
    {'Name':'Shit Happens', 'rating':6},
    {'Name':'Twister', 'rating':3},
    ]
# Ensure all favorite games are in the dataset
for game in played_games:
    if game['Name'] not in boardgames_df['Name'].values:
        raise ValueError(f"The game '{game['Name']}' is not found in the dataset.")


playedGames = pd.DataFrame(played_games)
inputId = boardgames_df[boardgames_df['Name'].isin(playedGames['Name'].tolist())]
playedGames = pd.merge(inputId,playedGames, on='Name')

userSubset = user_ratings_df[user_ratings_df['BGGId'].isin(playedGames['BGGId'].tolist())]
userSubsetGroup = userSubset.groupby(['Username'])
userSubsetGroup = sorted(userSubsetGroup, key=lambda x: len(x[1]), reverse=True)
userSubsetGroup = userSubsetGroup[0:100]

pearsonCorrelationDict = {}
#For every user group in our subset
for name, group in userSubsetGroup:
   #Let’s start by sorting the input and current user group so the values aren’t mixed up later on
   group = group.sort_values(by='BGGId')
   playedGames = playedGames.sort_values(by='BGGId')
   #Get the N for the formula
   nRatings = len(group)
   #Get the review scores for the movies that they both have in common
   temp_df = playedGames[playedGames['BGGId'].isin(group['BGGId'].tolist())]
   #And then store them in a temporary buffer variable in a list format to facilitate future calculations
   tempRatingList = temp_df['rating'].tolist()
   #Let’s also put the current user group reviews in a list format
   tempGroupList = group['Rating'].tolist()
   #Now let’s calculate the pearson correlation between two users, so called, x and y
   Sxx = sum([i**2 for i in tempRatingList]) - pow(sum(tempRatingList),2)/float(nRatings)
   Syy = sum([i**2 for i in tempGroupList]) - pow(sum(tempGroupList),2)/float(nRatings)
   Sxy = sum( i*j for i, j in zip(tempRatingList, tempGroupList)) - sum(tempRatingList)*sum(tempGroupList)/float(nRatings)
   #If the denominator is different than zero, then divide, else, 0 correlation.
   if Sxx != 0 and Syy != 0:
      pearsonCorrelationDict[name] = Sxy/sqrt(Sxx*Syy)
   else:
      pearsonCorrelationDict[name] = 0

pearsonDF = pd.DataFrame.from_dict(pearsonCorrelationDict, orient='index')
pearsonDF.columns = ['similarityIndex']
pearsonDF['Username'] = pearsonDF.index
pearsonDF.index = range(len(pearsonDF))

pearsonDF['Username'] = pearsonDF['Username'].apply(lambda x: x[0] if isinstance(x, tuple) else x)


topUsers=pearsonDF.sort_values(by='similarityIndex', ascending=False)[0:50]
print(topUsers.head())

topUsersRating=topUsers.merge(user_ratings_df, left_on='Username', right_on='Username', how='inner')
print(topUsersRating.head())

#Multiplies the similarity by the user’s ratings
topUsersRating['weightedRating'] = topUsersRating['similarityIndex']*topUsersRating['Rating']
print(topUsersRating.head())
#Applies a sum to the topUsers after grouping it up by userId
tempTopUsersRating = topUsersRating.groupby('BGGId').sum()[['similarityIndex','weightedRating']]
tempTopUsersRating.columns = ['sum_similarityIndex','sum_weightedRating']
print(tempTopUsersRating.head())

#Creates an empty dataframe
recommendation_df = pd.DataFrame()
#Now we take the weighted average
recommendation_df['weighted average recommendation score'] = tempTopUsersRating['sum_weightedRating']/tempTopUsersRating['sum_similarityIndex']
recommendation_df['BGGId'] = tempTopUsersRating.index
print(recommendation_df.head())

recommendation_df = recommendation_df.sort_values(by='weighted average recommendation score', ascending=False)
print(recommendation_df.head(10))

recommendation_df.reset_index(drop=True, inplace=True)

# Filter and create a DataFrame with the top 10 recommended games
top_recommendations = recommendation_df.head(10)
filtered_recommendations = top_recommendations[top_recommendations['BGGId'].isin(boardgames_df['BGGId'])]


final_recommendations = pd.merge(filtered_recommendations, boardgames_df, on='BGGId')

# Select only the required columns
final_recommendations = final_recommendations[['BGGId', 'Name', 'weighted average recommendation score']]

print(final_recommendations)