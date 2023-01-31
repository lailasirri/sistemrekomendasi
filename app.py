#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Data processing
import pandas as pd
import numpy as np
import scipy.stats 
# Visualization
import seaborn as sns

from flask import Flask, render_template, request

app = Flask(__name__, template_folder='templates')

# Read in data
ratings=pd.read_csv('rating.csv', sep=';')
# Read in data
hotels = pd.read_csv('hotel.csv', sep=';')

# Get the dataset information
ratings.info()

# Number of users
print('The ratings dataset has', ratings['userId'].nunique(), 'unique users')
# Number of movies
print('The ratings dataset has', ratings['hotelId'].nunique(), 'unique hotels')
# Number of ratings
print('The ratings dataset has', ratings['rating'].nunique(), 'unique ratings')
# List of unique ratings
print('The unique ratings are', sorted(ratings['rating'].unique()))

# Merge ratings and hotels datasets
df = pd.merge(ratings, hotels, on='hotelId', how='inner')

# Aggregate by hotel
agg_ratings = df.groupby('namahotel').agg(mean_rating = ('rating', 'mean'), number_of_ratings = ('rating', 'count')).reset_index()

# Keep the hotels with over 1 ratings
agg_ratings_1 = agg_ratings[agg_ratings['number_of_ratings']>1]
agg_ratings_1.info()

# Check popular hotels
agg_ratings_1.sort_values(by='number_of_ratings', ascending=False)

# Merge data
df_2 = pd.merge(df, agg_ratings_1[['namahotel']], on='namahotel', how='inner')
df_2.info()

# Number of users
print('The ratings dataset has', df_2['userId'].nunique(), 'unique users')
# Number of movies
print('The ratings dataset has', df_2['hotelId'].nunique(), 'unique hotels')
# Number of ratings
print('The ratings dataset has', df_2['rating'].nunique(), 'unique ratings')
# List of unique ratings
print('The unique ratings are', sorted(df_2['rating'].unique()))

# Create user-item matrix
matrix = df_2.pivot_table(index='namahotel', columns='userId', values='rating')

matrix1=pd.DataFrame(['matrix'])

file_name = 'matrix1.xlsx'
  
# saving the excel 
matrix.to_excel(file_name) 
print('DataFrame is written to Excel File successfully.')

# Normalize user-item matrix
matrix_norm = matrix.subtract(matrix.mean(axis=1), axis = 0)

# Item similarity matrix using Pearson correlation
item_similarity = matrix_norm.T.corr()

pearson=pd.DataFrame(['item_similarity'])

file_name = 'pearson.xlsx'
  
# saving the excel 
item_similarity.to_excel(file_name) 
print('DataFrame is written to Excel File successfully.')

# Pick a user ID
picked_userid = 1
# Pick a hotels
picked_hotel = 'ASTON Inn Mataram'
# Hotels that the target user has rating
picked_userid_rating = pd.DataFrame(matrix_norm[picked_userid].dropna(axis=0, how='all')                          .sort_values(ascending=False))                          .reset_index()                          .rename(columns={1:'rating1', 2: 'rating2', 3: 'rating3', 4: 'rating4'})
picked_userid_rating

# Similarity score of the movie American Pie with all the other movies
picked_hotel_similarity_score = item_similarity[[picked_hotel]].reset_index().rename(columns={'ASTON Inn Mataram':'similarity_score'})

similarity=pd.DataFrame(['picked_hotel_similarity_score'])

file_name = 'similarity.xlsx'
  
# saving the excel 
picked_hotel_similarity_score.to_excel(file_name) 
print('DataFrame is written to Excel File successfully.')

# Rank the similarities between the hotels user 1 rated and American Pie.
n = 5
picked_userid_rating_similarity = pd.merge(left=picked_userid_rating, 
                                            right=picked_hotel_similarity_score, 
                                            on='namahotel', 
                                            how='inner')\
                                     .sort_values('similarity_score', ascending=False)[:10]
# Take a look at the User 1 watched movies with highest similarity
picked_userid_rating_similarity

# Calculate the predicted rating using weighted average of similarity scores and the ratings from user 1
predicted_rating = round(np.average(picked_userid_rating_similarity['rating1'], 
                                    weights=picked_userid_rating_similarity['similarity_score']), 2)
print(f'The predicted rating for {picked_hotel} by user {picked_userid} is {predicted_rating}' )

# Item-based recommendation function
def item_based_rec(picked_userid=1, number_of_similar_items=3, number_of_recommendations =10):
  import operator
  # Hotels that the target user has not rating
  picked_userid_unrating = pd.DataFrame(matrix_norm[picked_userid].isna()).reset_index()
  picked_userid_unrating = picked_userid_unrating[picked_userid_unrating[1]==True]['namahotel'].values.tolist()
# Hotels that the target user has rating
  picked_userid_rating = pd.DataFrame(matrix_norm[picked_userid].dropna(axis=0, how='all')                            .sort_values(ascending=False))                            .reset_index()                            .rename(columns={1:'rating1', 2: 'rating2', 3: 'rating3', 4: 'rating4'})
  
  # Dictionary to save the unrating hoteland predicted rating pair
  rating_prediction ={}  
  # Loop through unrating hotels          
  for picked_hotel in picked_userid_unrating: 
    # Calculate the similarity score of the picked hotel with other hotels
    picked_hotel_similarity_score = item_similarity[[picked_hotel]].reset_index().rename(columns={picked_hotel:'similarity_score'})
    # Rank the similarities between the picked user rating hotel and the picked unrating hotel.
    picked_userid_rating_similarity = pd.merge(left=picked_userid_rating, 
                                                right=picked_hotel_similarity_score, 
                                                on='namahotel', 
                                                how='inner')\
                                        .sort_values('similarity_score', ascending=False)[:number_of_similar_items]
    # Calculate the predicted rating using weighted average of similarity scores and the ratings from user 1
    predicted_rating = round(np.average(picked_userid_rating_similarity['rating1'], 
                                        weights=picked_userid_rating_similarity['similarity_score']), 3)
# Save the predicted rating in the dictionary
    rating_prediction[picked_hotel] = predicted_rating
    # Return the top recommended movies
  return sorted(rating_prediction.items(), key=operator.itemgetter(1), reverse=True)[:number_of_recommendations]
# Get recommendations
recommended_hotel = item_based_rec(picked_userid=1, number_of_similar_items=3, number_of_recommendations =10)
recommended_hotel


@app.route('/', methods=['GET', 'POST'])
def main():
    if request.method == 'GET':
        return render_template('home.html', yin=yin)

    if request.method == 'POST':
        hotels = request.form['daftarhotel']
        res = item_based_rec(hotels)
        return render_template('background.html', result=res, yin=yin)
    else:
        return render_template('home.html')
 

if __name__ == '__main__':
    app.run(debug=True)

