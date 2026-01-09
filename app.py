import numpy as np
import tensorflow as tf
import streamlit as st
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import sequence

model=load_model("simple_rnn.h5")
# Load the IMDB dataset word index
word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}

# Step 2: Helper Functions
# Function to decode reviews
def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review])

# Function to preprocess user input
def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review

#prediction
def predict_sentiment(review):
    review=preprocess_text(review)
    pred=model.predict(review)
    sentiment="positive" if pred[0][0]>0.5 else "negitive"
    return sentiment,pred[0][0]

#streamlit app
st.title("IMDB Movie Review Sentiment Analysis using Simple RNN")
st.write("enter movie review for analysis")
user_input=st.text_area("Movie_Review")
if st.button("Classify"):
    sentiment,score=predict_sentiment(user_input)
    st.write(f'Sentiment: {sentiment}')
    st.write(f'Prediction Score: {score}')
else:
    st.write("enter movie review")
