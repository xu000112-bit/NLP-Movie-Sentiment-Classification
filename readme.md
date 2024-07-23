# Sentiment Review Detection

## Overview
This application utilizes Artificial Neural Networks (ANN), Recurrent Neural Networks (RNN), and Transformer models to detect and analyze sentiment in user-provided reviews. Users can input reviews manually or upload a CSV file for bulk analysis. The application is built using Streamlit, making it user-friendly for interactive prediction tasks.

## Features
- **Manual Review Input**: Users can type or paste a single review and get the sentiment analysis in real-time.
- **Bulk Review Processing**: Users can upload a CSV file containing multiple reviews to get batch sentiment predictions.
- **Model Selection**: Choose between ANN, RNN, and Transformer models for sentiment prediction.
- **Visual Analytics**: Generates bar plots showing the distribution of sentiments across the reviews.

## Prerequisites
Ensure you have the following installed:
- Python 3.8 or newer
- Streamlit
- TensorFlow
- Scikit-learn
- Pandas
- Numpy
- Matplotlib
- Joblib

## Usage

To run the application, navigate to the project directory in the terminal and run:

```bash
streamlit run main.py
```

### Interactive Components
- **Choose Model**: Select the prediction model from a dropdown.
- **Choose Mode**: Choose either 'Manual Input' for single review predictions or 'Upload CSV' for bulk predictions.
- **Detect Sentiment**: After entering a review or uploading a file, click this button to generate predictions.

## Data Format

For CSV uploads, ensure your data is formatted with a column named 'review' containing the text entries for analysis.

Example:
```csv
review
"I love this product!"
"Terrible customer service."
```

## Models

The models used in this application are trained using separate notebooks:
- `LSTMANDANN.ipynb` for the ANN and RNN models.
- `transformer.ipynb` for the Transformer model.

Ensure these models are correctly loaded from the `models` directory.
