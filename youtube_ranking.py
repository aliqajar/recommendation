import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import sys

# Synthetic data generation
np.random.seed(42)
num_users = 100
num_videos = 30
data = {
    'user_id': np.random.randint(1, num_users+1, size=1000),
    'video_id': np.random.randint(1, num_videos+1, size=1000),
    'watch_duration': np.random.rand(1000) * 100,
    'user_preference': np.random.rand(1000),
    'video_popularity': np.random.rand(1000)
}

df = pd.DataFrame(data)

# Candidate generation for user_id = 1
user_id = 1
user_data = df[df['user_id'] == user_id]
print("User Data Length:", len(user_data))

# Extract top candidate videos correctly
top_candidate_videos = user_data.sort_values(by='watch_duration', ascending=False).head(10)
print("Top Candidate Videos:", top_candidate_videos['video_id'].tolist())

# Extract video IDs correctly
video_ids = top_candidate_videos['video_id'].tolist()

# Feature preparation
features = df[df['video_id'].isin(video_ids)][['user_preference', 'video_popularity']]
labels = (df[df['video_id'].isin(video_ids)]['watch_duration'] > 50).astype(int)

print("Features Length:", len(features))
print("Labels Length:", len(labels))

if len(features) == 0:
    print("No data available for the selected videos. Check candidate generation and selection criteria.")
    sys.exit("Exiting due to empty dataset.")

# Split data
x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=42)

# Scaling features
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)


# Ranking model
model = LogisticRegression()
model.fit(x_train_scaled, y_train)

# Prediction and evaluation
y_pred_prob = model.predict_proba(x_test_scaled)[:, 1]
print("ROC AUC Score:", roc_auc_score(y_test, y_pred_prob))

# Post-processing
ranking_results = pd.DataFrame({
    'video_id': x_test.index,
    'predicted_probability': y_pred_prob
})

ranked_videos = ranking_results.sort_values(by='predicted_probability', ascending=False)

print('Top ranked videos for user:', user_id)
print(ranked_videos.head(5))
