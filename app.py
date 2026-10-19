import streamlit as st
import joblib
import re
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = re.sub(r'<.*?>', ' ', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    words = text.split()
    stop_words = set(stopwords.words('english')) - {'not', 'no', 'never', 'nor'}
    words = [w for w in words if w not in stop_words]
    return ' '.join(words)

model = joblib.load('sentiment_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

st.title("Movie Review Sentiment Analyzer")
st.write("Type a movie review below and see if it's predicted as Positive or Negative.")

user_input = st.text_area("Enter your review:")

if st.button("Analyze"):
    if user_input.strip() == "":
        st.warning("Please enter some text first.")
    else:
        cleaned = clean_text(user_input)
        vectorized = vectorizer.transform([cleaned])
        prediction = model.predict(vectorized)[0]

        if prediction == "positive":
            st.success(f"Prediction: {prediction.upper()} 😊")
        else:
            st.error(f"Prediction: {prediction.upper()} 😞")