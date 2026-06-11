import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend before importing pyplot

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import io
import os
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import mlflow
import numpy as np
import joblib
import re
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from mlflow.tracking import MlflowClient
import matplotlib.dates as mdates

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# ── Base directory ─────────────────────────────────────────────────────────
# app.py lives in:  .../yt-comment-sentiment-analysis-tool/flask_app/app.py
# vectorizer lives: .../yt-comment-sentiment-analysis-tool/tfidf_vectorizer.pkl
# So we go ONE level up from flask_app/ to reach the project root.
_FLASK_APP_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR       = os.path.abspath(os.path.join(_FLASK_APP_DIR, '..'))


# ── MLflow tracking URI ────────────────────────────────────────────────────
MLFLOW_TRACKING_URI = "http://ec2-13-218-87-51.compute-1.amazonaws.com:5000/"


def preprocess_comment(comment: str) -> str:
    """Apply preprocessing transformations to a comment."""
    try:
        comment = comment.lower()
        comment = comment.strip()
        comment = re.sub(r'\n', ' ', comment)
        comment = re.sub(r'[^A-Za-z0-9\s!?.,]', '', comment)

        stop_words = set(stopwords.words('english')) - {'not', 'but', 'however', 'no', 'yet'}
        comment = ' '.join([word for word in comment.split() if word not in stop_words])

        lemmatizer = WordNetLemmatizer()
        comment = ' '.join([lemmatizer.lemmatize(word) for word in comment.split()])

        return comment
    except Exception as e:
        app.logger.error(f"Error in preprocessing comment: {e}")
        return comment


def load_model_and_vectorizer(model_name: str, stage: str, vectorizer_path: str):
    """
    Load model from MLflow Model Registry by STAGE (not version number)
    and vectorizer from an absolute local path.
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    #  – load by stage so the API always gets the right promoted model
    model_uri = f"models:/{model_name}/{stage}"
    model = mlflow.pyfunc.load_model(model_uri)

    # – use the absolute path so CWD doesn't matter
    vectorizer = joblib.load(vectorizer_path)
    return model, vectorizer


# Resolve vectorizer path relative to this file
VECTORIZER_PATH = os.path.join(BASE_DIR, 'tfidf_vectorizer.pkl')

# Initialise model and vectorizer at startup
model, vectorizer = load_model_and_vectorizer(
    model_name="my_model",
    stage="Staging",           # – use stage, not hardcoded version "1"
    vectorizer_path=VECTORIZER_PATH
)


def transform_to_dataframe(comments: list) -> pd.DataFrame:
    """
    Preprocess → TF-IDF transform → return a pandas DataFrame.

    mlflow.pyfunc.load_model wraps sklearn models and expects a
    DataFrame (matching the signature inferred during mlflow.sklearn.log_model).
    Passing a raw sparse matrix causes a type-mismatch error.
    """
    preprocessed = [preprocess_comment(c) for c in comments]
    sparse_matrix = vectorizer.transform(preprocessed)
    feature_names = vectorizer.get_feature_names_out()
    return pd.DataFrame(sparse_matrix.toarray(), columns=feature_names)


@app.route('/')
def home():
    return "Welcome to the Sentiment Analysis Flask API"


@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    comments = data.get('comments')

    if not comments:
        return jsonify({"error": "No comments provided"}), 400

    try:
        #  – pass DataFrame, not sparse matrix
        input_df = transform_to_dataframe(comments)
        predictions = model.predict(input_df).tolist()

        #  – keep predictions as int for consistency; convert to str only at
        # the serialisation boundary so downstream code can rely on numeric values
        predictions = [int(pred) for pred in predictions]
    except Exception as e:
        app.logger.error(f"Prediction failed: {e}")
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

    response = [
        {"comment": comment, "sentiment": sentiment}
        for comment, sentiment in zip(comments, predictions)
    ]
    return jsonify(response)


@app.route('/predict_with_timestamps', methods=['POST'])
def predict_with_timestamps():
    data = request.json
    comments_data = data.get('comments')

    if not comments_data:
        return jsonify({"error": "No comments provided"}), 400

    try:
        comments   = [item['text']      for item in comments_data]
        timestamps = [item['timestamp'] for item in comments_data]

        #  – pass DataFrame, not sparse matrix
        input_df = transform_to_dataframe(comments)
        predictions = model.predict(input_df).tolist()

        #  – keep as int; the trend graph casts back to int anyway
        predictions = [int(pred) for pred in predictions]
    except Exception as e:
        app.logger.error(f"Prediction failed: {e}")
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

    response = [
        {"comment": comment, "sentiment": sentiment, "timestamp": timestamp}
        for comment, sentiment, timestamp in zip(comments, predictions, timestamps)
    ]
    return jsonify(response)


@app.route('/generate_chart', methods=['POST'])
def generate_chart():
    try:
        data = request.get_json()
        sentiment_counts = data.get('sentiment_counts')

        if not sentiment_counts:
            return jsonify({"error": "No sentiment counts provided"}), 400

        #  – accept both int and string keys from the frontend
        labels = ['Positive', 'Neutral', 'Negative']
        sizes = [
            int(sentiment_counts.get(1,  sentiment_counts.get('1',  0))),
            int(sentiment_counts.get(0,  sentiment_counts.get('0',  0))),
            int(sentiment_counts.get(-1, sentiment_counts.get('-1', 0))),
        ]

        if sum(sizes) == 0:
            return jsonify({"error": "Sentiment counts sum to zero"}), 400

        colors = ['#36A2EB', '#C9CBCF', '#FF6384']

        plt.figure(figsize=(6, 6))
        plt.pie(
            sizes,
            labels=labels,
            colors=colors,
            autopct='%1.1f%%',
            startangle=140,
            textprops={'color': 'w'}
        )
        plt.axis('equal')

        img_io = io.BytesIO()
        plt.savefig(img_io, format='PNG', transparent=True)
        img_io.seek(0)
        plt.close()

        return send_file(img_io, mimetype='image/png')
    except Exception as e:
        app.logger.error(f"Error in /generate_chart: {e}")
        return jsonify({"error": f"Chart generation failed: {str(e)}"}), 500


@app.route('/generate_wordcloud', methods=['POST'])
def generate_wordcloud():
    try:
        data = request.get_json()
        comments = data.get('comments')

        if not comments:
            return jsonify({"error": "No comments provided"}), 400

        preprocessed_comments = [preprocess_comment(c) for c in comments]
        text = ' '.join(preprocessed_comments)

        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color='black',
            colormap='Blues',
            stopwords=set(stopwords.words('english')),
            collocations=False
        ).generate(text)

        img_io = io.BytesIO()
        wordcloud.to_image().save(img_io, format='PNG')
        img_io.seek(0)

        return send_file(img_io, mimetype='image/png')
    except Exception as e:
        app.logger.error(f"Error in /generate_wordcloud: {e}")
        return jsonify({"error": f"Word cloud generation failed: {str(e)}"}), 500


@app.route('/generate_trend_graph', methods=['POST'])
def generate_trend_graph():
    try:
        data = request.get_json()
        sentiment_data = data.get('sentiment_data')

        if not sentiment_data:
            return jsonify({"error": "No sentiment data provided"}), 400

        df = pd.DataFrame(sentiment_data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)

        #  – cast to int reliably whether values arrived as str or int
        df['sentiment'] = df['sentiment'].astype(int)

        sentiment_labels = {-1: 'Negative', 0: 'Neutral', 1: 'Positive'}

        monthly_counts = df.resample('ME')['sentiment'].value_counts().unstack(fill_value=0)
        monthly_totals = monthly_counts.sum(axis=1)
        monthly_percentages = (monthly_counts.T / monthly_totals).T * 100

        for sv in [-1, 0, 1]:
            if sv not in monthly_percentages.columns:
                monthly_percentages[sv] = 0

        monthly_percentages = monthly_percentages[[-1, 0, 1]]

        plt.figure(figsize=(12, 6))

        colors = {-1: 'red', 0: 'gray', 1: 'green'}
        for sv in [-1, 0, 1]:
            plt.plot(
                monthly_percentages.index,
                monthly_percentages[sv],
                marker='o',
                linestyle='-',
                label=sentiment_labels[sv],
                color=colors[sv]
            )

        plt.title('Monthly Sentiment Percentage Over Time')
        plt.xlabel('Month')
        plt.ylabel('Percentage of Comments (%)')
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=12))
        plt.legend()
        plt.tight_layout()

        img_io = io.BytesIO()
        plt.savefig(img_io, format='PNG')
        img_io.seek(0)
        plt.close()

        return send_file(img_io, mimetype='image/png')
    except Exception as e:
        app.logger.error(f"Error in /generate_trend_graph: {e}")
        return jsonify({"error": f"Trend graph generation failed: {str(e)}"}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)