# Movie Review Sentiment Analyzer

A Streamlit web app that predicts whether a movie review is **positive** or **negative**.
Trained on the [IMDB 50K Movie Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews) dataset.

## How it works

1. **Cleaning** – strips HTML tags, punctuation and numbers, lowercases the text, and removes English stopwords (keeping negations like *not*, *no*, *never*).
2. **Features** – TF-IDF vectorizer (top 15,000 features, 1–3 word n-grams).
3. **Model** – Logistic Regression (scikit-learn), ~90% accuracy on the held-out test set.

## Project files

| File | Purpose |
|------|---------|
| `app.py` | Streamlit app |
| `Untitled.ipynb` | Notebook: download data, clean, train and evaluate the model |
| `sentiment_model.pkl` | Trained Logistic Regression model |
| `tfidf_vectorizer.pkl` | Fitted TF-IDF vectorizer |

## Run the app

```bash
pip install streamlit scikit-learn nltk joblib
python -c "import nltk; nltk.download('stopwords')"
streamlit run app.py
```

Then open the URL Streamlit prints (usually http://localhost:8501), type a review and click **Analyze**.

## Retrain the model

Open `Untitled.ipynb`, put your own Kaggle username and API key in the first cell, and run all cells. The dataset is downloaded automatically and is not stored in this repo.
