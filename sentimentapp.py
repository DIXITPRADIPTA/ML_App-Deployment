import re
import pickle
import streamlit as st
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
import base64

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

vocab = pickle.load(open("vocab.pkl", 'rb'))
model = pickle.load(open("LR1_model.pkl", 'rb'))


stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

def preprocess(ReviewText, flag):
    # Removing special characters and digits
    sentence = re.sub("[^a-zA-Z]", " ", ReviewText)

    # change sentence to lower case
    sentence = sentence.lower()

    # tokenize into words
    tokens = word_tokenize(sentence)

    # remove stop words
    clean_tokens = [t for t in tokens if t not in stopwords.words("english")]

    # Stemming/Lemmatization
    if flag == 'stem':
        clean_tokens = [stemmer.stem(word) for word in clean_tokens]
    else:
        clean_tokens = [lemmatizer.lemmatize(word) for word in clean_tokens]

    return " ".join(clean_tokens)


def predict_sentiment(review):
    preprocessed_review = preprocess(review, 'lemmatize')
    vectorized_review = vocab.transform([preprocessed_review])
    prediction = model.predict(vectorized_review)
    st.success('Prediction is done successfully!', icon="✅")
    if prediction[0] == 1:
        sentiment = 'Positive \U0001F600'
    else:
        sentiment = 'Negative \U0001F614'
    return sentiment


def main():
    page_bg_img="""
    
    <style>
    [data-testid="stAppViewContainer"] {
    background-image: url(https://wallpaperaccess.com/full/38144.jpg);
    background-size: cover;
    }
    [data-testid="stHeader"] {
    background-color: rgba(0, 0, 0, 0);
    }

    [data-testid="stToolbar"] {
    right:2rem;
    }
    </style>
    """
    st.markdown(page_bg_img,unsafe_allow_html=True)

     
    st.title('Sentiment Analysis App')
    review = st.text_input('Enter your review:')
    
    if st.button('Predict'):
        if review.strip() == '':
            st.error('Please enter a review.')
        else:
            sentiment = predict_sentiment(review)
            st.write('Sentiment:', sentiment)


if __name__ == '__main__':
    main()