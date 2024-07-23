import streamlit as st
import pandas as pd
import joblib
from tensorflow.keras.models import load_model
import os
from datetime import datetime
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import joblib
import json
from keras.preprocessing.text import tokenizer_from_json
import matplotlib.pyplot as plt

HISTORY_DIR = 'history'
# Ensure history directory exists
if not os.path.exists(HISTORY_DIR):
    os.makedirs(HISTORY_DIR)


# Load the trained models
ann_model = load_model("models\ANN_IMDB.h5")
rnn_model = load_model("models\LSTM_IMDB.keras")


def preprocess_data_single(data, model_choice):
    if model_choice == "ANN":
        # For ANN, using TfidfVectorizer
        tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')
        preprocessed_data = tfidf_vectorizer.transform([data]).toarray()
    elif model_choice == "RNN":
        # For RNN, using Tokenizer
        # Load the pre-fitted Tokenizer
        with open('tokenizer.json') as f:
            tokenizer_data = json.load(f)
            tokenizer = tokenizer_from_json(tokenizer_data)

        sequences = tokenizer.texts_to_sequences([data])
        preprocessed_data = pad_sequences(sequences, maxlen=500)  # Use the same max length as in training
        return preprocessed_data
    # Add an else if block for the transformer model if applicable
    return preprocessed_data

def preprocess_data_csv(data, model_choice):
    if model_choice == "ANN":
        # For ANN, using TfidfVectorizer
        tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')
        preprocessed_data = tfidf_vectorizer.transform(data['review']).toarray()
    elif model_choice == "RNN":
        # For RNN, using Tokenizer
        tokenizer = Tokenizer(num_words=5000)
        tokenizer.fit_on_texts(data['review'])
        sequences = tokenizer.texts_to_sequences(data['review'])
        preprocessed_data = pad_sequences(sequences, maxlen=500) # Assuming max_review_length = 500
    # Add an else if block for the transformer model if applicable
    return preprocessed_data

def predict_sentiment(data, model_choice, mode):
    if mode == "Manual Input":
        # Prediction logic for a single review
        if model_choice == "ANN":
            preprocessed_data = preprocess_data_single(data, model_choice)
            prediction = ann_model.predict(preprocessed_data)
        elif model_choice == "RNN":
            preprocessed_data = preprocess_data_single(data, model_choice)
            prediction = rnn_model.predict(preprocessed_data)
        # Include logic for the transformer model if available
        return prediction
    elif mode == "Upload CSV":
        preprocessed_data = preprocess_data_csv(data, model_choice)
        # Prediction logic for CSV data
        if model_choice == "ANN":
            predictions = ann_model.predict(preprocessed_data)
        elif model_choice == "RNN":
            predictions = rnn_model.predict(preprocessed_data)
        # Include logic for the transformer model if available
        return predictions

# Function to save data and return count for visualization
def save_and_visualize(data, predictions):
    # Save the uploaded data for history with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    data['Prediction'] = ['Positive' if pred[0] > 0.5 else 'Negative' for pred in predictions]
    data.to_csv(os.path.join(HISTORY_DIR, f"data_{timestamp}.csv"), index=False)

    # Create a bar plot for sentiment count
    sentiment_count = data['Prediction'].value_counts()
    plt.figure(figsize=(8, 4))
    sentiment_count.plot(kind='bar')
    plt.title('Sentiment Count')
    plt.xlabel('Sentiment')
    plt.ylabel('Count')
    st.pyplot(plt)


st.title("Sentiment Reviews Detection")

model_choice = st.selectbox("Choose Model", ["ANN", "RNN", "TRANSFORMER"])

# Choose mode
mode = st.selectbox("Choose Mode", ["Manual Input", "Upload CSV"])

if mode == "Manual Input":
    review_input = st.text_area("Enter Review Text")
    if st.button("Detect Sentiment"):
        prediction = predict_sentiment(review_input, model_choice, mode)
        # Convert prediction to a value between 0 and 1
        prediction_value = prediction[0][0]
        st.write("Prediction: ", "Positive" if prediction_value > 0.5 else "Negative")
        # Create a slider to display the prediction value
        st.slider("Confidence", min_value=0.0, max_value=1.0, value=prediction_value, format="%.2f")


elif mode == "Upload CSV":
    uploaded_file = st.file_uploader("Upload CSV file", type="csv")
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        predictions = predict_sentiment(data, model_choice, mode)
        # Displaying predictions with expanders for each review
        for i, (review, pred) in enumerate(zip(data['review'], predictions)):
            sentiment = 'Positive' if pred[0] > 0.5 else 'Negative'
            with st.expander(f"Review {i + 1}: {sentiment}"):
                st.write(review)

        save_and_visualize(data, predictions)
