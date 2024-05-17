
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MultiLabelBinarizer


class WatchTimePredictor:

    def __init__(self):
        self.model = None


    def load_data(self, file_path):
        data = pd.read_csv(file_path)
        return data
    

    def preprocess_data(self, data):
        # prepocessing such as handling missign values, convert categorical values etc
        preprocessed_data = data.copy()

        # one hot encoding on the gender
        preprocessed_data = pd.get_dummies(preprocessed_data, columns=['gender'])

        # convert genre to binary columns
        mlb = MultiLabelBinarizer()
        genre_columns = mlb.fit_transform(preprocessed_data['genre_preferences'].str.split(','))
        genre_df = pd.DataFrame(genre_columns, columns=mlb.classes_, index=preprocessed_data.index)
        preprocessed_data = pd.concat([preprocessed_data, genre_df], axis=1)
        preprocessed_data.drop('genre_preferences', axis=1, inplace=True)

        return preprocessed_data
    

    def split_data(self, data):
        print(data.columns)
        # split data into features and targets
        x = data.drop('watch_time', axis=1)
        y = data['watch_time']

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
        return x_train, x_test, y_train, y_test
    

    def train_model(self, x_train, y_train):
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(x_train, y_train)
        self.model = model


    def evaluate_model(self, x_test, y_test):
        y_pred = self.model.predict(x_test)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)

        print(f'Mean Absolue Error:{mae: .2f}')
        print(f'Mean Squared Error: {mse: .2f}')


    def predict(self, user_data):
        watch_time = self.model.predict(user_data)
        return watch_time
    


predictor = WatchTimePredictor()

data = predictor.load_data('watch_data.csv')
preprocessed_data = predictor.preprocess_data(data)

x_train, x_test, y_train, y_test = predictor.split_data(preprocessed_data)

predictor.train_model(x_train, y_train)

predictor.evaluate_model(x_test, y_test)

new_user_data = pd.DataFrame({
    'user_id': [12345],
    'age': [28],
    'gender_Female': [0],
    'gender_Male': [1],
    'Action': [1],
    'Comedy': [0],
    'Documentary': [0],
    'Drama': [0],
    'Romance': [0],
    'Sci-Fi': [0],
    'Thriller': [0]
})


predicted_watch_time = predictor.predict(new_user_data)
print(f'Predicted watch time for the new user: {predicted_watch_time[0]: .2f} minutes')


