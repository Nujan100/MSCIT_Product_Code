from flask import Flask, render_template, request
import pandas as pd
from surprise import KNNWithMeans, Dataset, Reader, accuracy
from surprise.model_selection import train_test_split

app = Flask(__name__)

# Load data
df = pd.read_csv('G:/MscIT/DataSet/amazon_product_reviews/ratings_Electronics.csv', names=['userId', 'productId', 'rating', 'timestamp'])


# Keep a subset of data for faster testing
sample_size = 5000
data_sample = df.sample(sample_size)

# Surprise library setup
reader = Reader(rating_scale=(1, 5))
surprise_data = Dataset.load_from_df(data_sample[['userId', 'productId', 'rating']], reader)
trainset, testset = train_test_split(surprise_data, test_size=0.3, random_state=42)

# Collaborative filtering setup
algo = KNNWithMeans(k=5, sim_options={'name': 'pearson', 'user_based': False})
algo.fit(trainset)
test_pred = algo.test(testset)

# Calculate RMSE and MAE
rmse = accuracy.rmse(test_pred, verbose=False)
mae = accuracy.mae(test_pred, verbose=False)

# Get recommendations for a new product
def get_recommendations(new_product):
    recommendations = algo.get_neighbors(algo.trainset.to_inner_iid(new_product), k=20)
    recommended_products = [algo.trainset.to_raw_iid(inner_id) for inner_id in recommendations]
    return recommended_products

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        new_product = request.form['new_product']
        recommendations = get_recommendations(new_product)
        return render_template('recommendation.html', new_product=new_product, recommendations=recommendations)
    except Exception as e:
        error_message = str(e)
        return render_template('error.html', error_message=error_message)
    
def load_dataset():
    try:
        # Adjust the file path based on your directory structure
        df = pd.read_csv('path/to/your/dataset.csv', names=['userId', 'productId', 'rating', 'timestamp'])
        all_data = df.sample(n=1564896, ignore_index=True)
        all_data.drop('timestamp', axis=1, inplace=True)

        # Add any additional preprocessing steps if needed

        return all_data

    except Exception as e:
        raise e

# Your existing code...

if __name__ == '__main__':
    app.run(debug=True)
