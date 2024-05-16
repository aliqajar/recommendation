import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

data = {'user_id': [1, 1, 1, 2, 2, 3, 3, 4],
        'item_id': [101, 102, 105, 101, 104, 102, 103, 105],
        'rating': [5, 3, 2, 5, 4, 4, 5, 3]}


df = pd.DataFrame(data)
print (df)

user_item_matrix = df.pivot_table(index='user_id', columns='item_id', values='rating').fillna(0)
print('User Item Matrix')
print(user_item_matrix)

user_similarity = cosine_similarity(user_item_matrix)
print(user_similarity)

user_similarity_df = pd.DataFrame(user_similarity, index=user_item_matrix.index, columns=user_item_matrix.index)
print('User Similarity Matrix:')
print(user_similarity_df)


def recommend_items(user_item_matrix, user_similarity_df, k=3):

    user_item_scores = user_similarity_df.dot(user_item_matrix)
    print(user_item_scores)

    user_item_scores[user_item_matrix > 0] = 0
    print(user_item_scores)

    top_items = user_item_scores.apply(lambda x: pd.Series(x.nlargest(k).index), axis=1)
    print(top_items)

    return top_items



k=2
recommendations = recommend_items(user_item_matrix, user_similarity_df, k)



