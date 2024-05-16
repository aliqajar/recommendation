

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix



data = {'user_id': [1, 1, 1, 2, 2, 3, 3, 4],
        'item_id': [101, 102, 103, 101, 104, 102, 103, 104],
        'rating': [5, 3, 2, 5, 4, 4, 5, 3]}

df = pd.DataFrame(data)

user_item_matrix = df.pivot(index='user_id', columns='item_id', values='rating').fillna(0)
print(user_item_matrix)

user_similarity = cosine_similarity(user_item_matrix)
user_similarity_df = pd.DataFrame(user_similarity, index=user_item_matrix.index, columns=user_item_matrix.index)
print(user_similarity_df)


# get top similar users
def get_top_similar_users(user_id, similarity_matrix, k=3):
    similar_users = similarity_matrix[user_id].sort_values(ascending=False)[1:k+1].index
    return similar_users


# function to recommend items for a user
def recommend_items(user_id, userr_item_matrix, similarity_matrix, k=3):
    # get top similar users
    similar_users = get_top_similar_users(user_id, similarity_matrix, k)
    print('similar users', similar_users)

    # get the items rated by similar users
    similar_user_items = user_item_matrix.loc[similar_users]
    print('similar user items', similar_user_items)

    # calculate the average rating or each item
    item_ratings = similar_user_items.mean(axis=0)
    print('item ratings', item_ratings)

    # recommend items with the highest average rating that the usr hasn't rated yet
    user_rated_items = user_item_matrix.loc[user_id].to_numpy().nonzero()[0]
    print(user_rated_items)

    user_rated_items_labels = user_item_matrix.columns[user_rated_items].tolist()
    print(user_rated_items_labels)

    recommended_items = item_ratings.drop(labels = user_rated_items_labels, errors='ignore').sort_values(ascending=False).index
    print(recommended_items)

    return recommended_items



user_id = 1
k = 2

recommended_items = recommend_items(user_id, user_item_matrix, user_similarity_df, k)
print(f'Recommended items for user {user_id}: {recommended_items}')




