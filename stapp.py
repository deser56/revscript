

import streamlit as st
import unittest
import string
import streamlit as st
import spacy
import seaborn as sns
import re
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import nltk
import matplotlib.pyplot as plt
import logging
import json
import joblib
import hashlib

from wordcloud import WordCloud
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding, SpatialDropout1D
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from imblearn.over_sampling import SMOTE
from datetime import datetime
from io import StringIO







nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('vader_lexicon')
nltk.download('punkt', download_dir='/content')
nltk.download('stopwords', download_dir='/content')
nltk.download('wordnet', download_dir='/content')
nltk.download('vader_lexicon', download_dir='/content')
nltk.data.path.append('/content')

@st.cache_resource
def download_punkt():
    nltk.download('punkt')



# Download NLTK resources at startup
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', force=True)

try:
    nltk.download('vader_lexicon', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
except Exception as e:
    st.error(f"Error downloading NLTK resources: {str(e)}")

# Define stopwords and lemmatizer
#stop_words = set(stopwords.words('english'))
stop_words = stopwords.words('english')
stop_words += ['…', 'nuclearenergy', '’', 'yes', '核エネルギーの潜在的な危険性は、いくら強調してもし過ぎるということはない。']
lemmatizer = WordNetLemmatizer()

# Security configurations
class SecurityConfig:
    def __init__(self):
        self.ALLOWED_EXTENSIONS = {'csv', 'txt'}
        self.MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB

    def validate_file(self, file):
        if not file:
            return False
        extension = file.name.split('.')[-1].lower()
        return extension in self.ALLOWED_EXTENSIONS and len(file.getvalue()) <= self.MAX_FILE_SIZE

# text processing and analysis
class TextAnalyzer:
    def __init__(self):
        try:
            self.sia = SentimentIntensityAnalyzer()
        except Exception as e:
            st.error(f"Error initializing VADER analyzer: {str(e)}")
            self.sia = None

    def analyze_sentiment(self, text):
        results = {}

        # VADER sentiment
        if self.sia:
            try:
                results['vader'] = self.sia.polarity_scores(str(text))['compound']
            except Exception as e:
                results['vader'] = 0
                logging.error(f"VADER analysis error: {str(e)}")

        # TextBlob sentiment
        try:
            results['textblob'] = TextBlob(str(text)).sentiment.polarity
        except Exception as e:
            results['textblob'] = 0
            logging.error(f"TextBlob analysis error: {str(e)}")

        return results

    def get_basic_stats(self, text):
        """Get basic text statistics"""
        text = str(text)
        return {
            'char_count': len(text),
            'word_count': len(text.split()),
            'sentence_count': len(text.split('.')),
        }

# Data processing and saving utilities
class DataManager:
    @staticmethod
    def save_results(data, filename):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"analysis_results_{timestamp}.json"
        with open(filename, 'w') as f:
            json.dump(data, f)
        return filename


def main():
    st.set_page_config(page_title="Text Analysis Pipeline", layout="wide")
    st.title("Sentimental Data Analysis Pipeline")
    st.sidebar.title("Data Pipeline Steps")

    # Initialize components
    security = SecurityConfig()
    analyzer = TextAnalyzer()

    # Setup logging
    logging.basicConfig(filename='app.log', level=logging.INFO)

    # Simple authentication (accept any password)
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False

    # Show password input only if not authenticated
    if not st.session_state.authenticated:
        password = st.text_input("Enter password", type="password")
        if password:
            st.session_state.authenticated = True
            st.success("Authenticated")

    # Main navigation triggered after authentication
    if st.session_state.authenticated:
        pages = {
            "Data Loading": render_data_loading,
            "Data Display": render_data_display,
            "Data Cleaning": render_data_cleaning,
            "Data Preprocessing": render_data_preprocessing,
            "Data Analysis Methods and Visualization": render_analysis,
            "Model Training and Evaluation": render_model_training,
            "Generate Reports": render_reports,
            "Help": render_help
        }

        page = st.sidebar.radio("Navigation", list(pages.keys()))
        pages[page](st.session_state, analyzer, security)

def render_data_loading(session_state, analyzer, security):
    st.title("Data Loading")  # Main heading for the page
    st.subheader("Choose data source")

    # Add options for data input
    data_source = st.radio("Choose data source", ["Upload File", "Text Input", "URL Input"])

    # Option 1: Upload File
    if data_source == "Upload File":
        uploaded_file = st.file_uploader("Upload CSV/TXT file", type=["csv", "txt"])
        if uploaded_file and security.validate_file(uploaded_file):
            try:
                data = pd.read_csv(uploaded_file)
                session_state.data = data
                st.success("File uploaded successfully!")
                st.write(data.head())  # Display a preview of the data
            except Exception as e:
                st.error(f"Error processing the file: {str(e)}")
        elif uploaded_file:
            st.error("Invalid file. Please upload a valid CSV or TXT file.")

    # Option 2: Text Input
    elif data_source == "Text Input":
        text_input = st.text_area("Enter text data for analysis")
        if text_input:
            try:
                session_state.data = pd.DataFrame([{"text": text_input}])
                st.success("Text input received!")
                st.write(session_state.data)  # Display a preview of the data
            except Exception as e:
                st.error(f"Error processing text input: {str(e)}")

    # Option 3: URL Input
    elif data_source == "URL Input":
        url = st.text_input("Enter URL")
        if url and st.button("Fetch Data"):
            try:
                data = pd.read_csv(url)
                session_state.data = data
                st.success("Data fetched successfully from the URL!")
                st.write(data.head())  # Display a preview of the data
            except Exception as e:
                st.error(f"Failed to fetch data from URL: {str(e)}")

def render_data_display(session_state, analyzer, security):
    st.title("Data Display")

    # Check if data is loaded
    if 'data' in session_state and session_state.data is not None:
        # Radio buttons for display options
        display_option = st.radio(
            "Choose display format",
            ["Raw Data", "Summary Statistics", "Column Info","Summary Of Missing Values","Summary Of Duplicates"]
        )

        # Display raw data
        if display_option == "Raw Data":
            st.dataframe(session_state.data)

        # Display summary statistics
        elif display_option == "Summary Statistics":
            st.write(session_state.data.describe())

        # Display column information
        elif display_option == "Column Info":
            buffer = StringIO()
            session_state.data.info(buf=buffer)
            info_output = buffer.getvalue()
            st.text(info_output)

        # Display count of missing values in each column
        elif display_option == "Summary Of Missing Values":
            missing_values = session_state.data.isna().sum()
            st.write(missing_values)

        # Display count of duplicate values
        elif display_option == "Summary Of Duplicates":
            duplicate_values = session_state.data.duplicated().sum()
            st.write(f"Number of duplicate rows: {duplicate_values}")

    else:
        # Warning if no data is loaded
        st.warning("Please load data first.")

def render_data_cleaning(session_state, analyzer, security):
    st.title("Data Cleaning")

    # Check if data is loaded
    if 'data' in session_state and session_state.data is not None:
        # Create a backup of the original data if not already present
        if 'original_data' not in session_state:
            session_state.original_data = session_state.data.copy()

        # Select cleaning operation
        cleaning_option = st.selectbox(
            "Select cleaning operation",
            ["Handle Missing Values", "Remove Duplicates", "Filter Data"]
        )

        # Handle Missing Values
        if cleaning_option == "Handle Missing Values":
            temp_data = session_state.data.copy()  # Temporary DataFrame for previewing changes
            st.write("Data with Missing Values:")
            st.write(temp_data[temp_data.isnull().any(axis=1)].head())  # Display rows with missing values

            # Track columns to drop missing values in one step
            columns_to_drop = []

            # Handle missing values column by column
            for column in temp_data.columns:
                if temp_data[column].isnull().any():
                    st.write(f"Column with missing values: {column}")
                    method = st.radio(
                        f"Choose how to handle missing values in {column}:",
                        ["Drop", "Fill with mean", "Fill with median", "Fill with mode"],
                        key=column  # Ensure unique keys for each column
                    )

                    # Apply selected method to the temporary DataFrame
                    if method == "Drop":
                        columns_to_drop.append(column)
                    elif method == "Fill with mean" and pd.api.types.is_numeric_dtype(temp_data[column]):
                        temp_data[column].fillna(temp_data[column].mean(), inplace=True)
                    elif method == "Fill with median" and pd.api.types.is_numeric_dtype(temp_data[column]):
                        temp_data[column].fillna(temp_data[column].median(), inplace=True)
                    elif method == "Fill with mode":
                        temp_data[column].fillna(temp_data[column].mode()[0], inplace=True)

            # Drop rows with missing values for selected columns once after the loop
            if columns_to_drop:
                temp_data.dropna(subset=columns_to_drop, inplace=True)

            # Display a preview of the cleaned data
            st.write("Preview of Cleaned Data:")
            st.write(temp_data.head())

            # Provide confirmation buttons
            if st.button("Confirm Changes"):
                if not temp_data.empty:
                    session_state.data = temp_data
                    st.success("Changes applied successfully!")
                else:
                    st.error("All data would be removed. Please adjust your cleaning options.")

            if st.button("Undo Changes"):
                session_state.data = session_state.original_data.copy()
                st.success("Changes reverted to original data!")

        # Remove Duplicates
        elif cleaning_option == "Remove Duplicates":
            st.write(f"Current data shape: {session_state.data.shape}")
            if st.button("Remove Duplicate Rows"):
                session_state.data = session_state.data.drop_duplicates()
                st.success("Duplicate rows removed!")
                st.write(f"New data shape: {session_state.data.shape}")

        # Filter Data
        elif cleaning_option == "Filter Data":
            column = st.selectbox("Select column to filter", session_state.data.columns)
            if pd.api.types.is_numeric_dtype(session_state.data[column]):
                min_val, max_val = st.slider(
                    "Filter range",
                    float(session_state.data[column].min()),
                    float(session_state.data[column].max()),
                    (float(session_state.data[column].min()), float(session_state.data[column].max()))
                )
                temp_data = session_state.data[
                    (session_state.data[column] >= min_val) & (session_state.data[column] <= max_val)
                ]

                st.write("Filtered Data Preview:")
                st.write(temp_data.head())

                if st.button("Apply Filter"):
                    if not temp_data.empty:
                        session_state.data = temp_data
                        st.success("Filter applied successfully!")
                    else:
                        st.error("Filter resulted in no data. Please adjust your filter criteria.")
            else:
                st.warning("Selected column is not numeric. Filtering is only available for numeric columns.")

    else:
        # Warning if no data is loaded
        st.warning("Please load data first.")

# Load SpaCy model
nlp = spacy.load("en_core_web_sm")

def render_data_preprocessing(session_state, analyzer, security):
    st.title("Data Preprocessing")

    # Check if data is loaded
    if 'data' in session_state and session_state.data is not None:
        # Create a backup of the original data if not already present
        if 'preprocessing_backup' not in session_state:
            session_state.preprocessing_backup = session_state.data.copy()

        # Step 0: Allow the user to select the text column
        st.subheader("Select Text Column for Preprocessing")
        text_columns = session_state.data.select_dtypes(include=['object', 'string']).columns
        if len(text_columns) == 0:
            st.error("No text columns found in the dataset. Please upload a dataset with text data.")
            return

        # Dropdown to select text column
        selected_column = st.selectbox("Choose the column containing text data:", text_columns)
        st.write(f"Preview of selected column: {selected_column}")
        st.write(session_state.data[selected_column].head())

        # Step 1: Remove Irrelevant Elements
        if st.button("Remove Irrelevant Elements"):
            def clean_text(text):
                # Ensure input is a string
                if not isinstance(text, str):
                    text = str(text) if text is not None else ""
                # Remove URLs, mentions, hashtags, punctuation, and special characters
                text = re.sub(r"http\S+|www\S+|https\S+|\@\w+|\#|[^\w\s]", '', text)
                # Remove words with numbers and extra spaces
                text = re.sub(r'\w*\d\w*', ' ', text)
                text = re.sub('\s+', ' ', text).strip()
                return text.lower()

            # Apply the cleaning function
            cleaned_column = "cleaned_text"
            session_state.data[cleaned_column] = session_state.data[selected_column].apply(clean_text)

            st.success("Irrelevant elements removed successfully!")
            st.write(session_state.data[[selected_column, cleaned_column]].head())

        # Step 2: Tokenization
        if st.button("Tokenize Text"):
            def spacy_tokenizer(text):
                doc = nlp(text)
                return [token.text for token in doc if not token.is_space]

            cleaned_column = "cleaned_text"
            if cleaned_column in session_state.data.columns:
                session_state.data["tokens"] = session_state.data[cleaned_column].apply(spacy_tokenizer)

                st.success("Text tokenized successfully!")
                st.write(session_state.data[["tokens"]].head())
            else:
                st.error(f"'{cleaned_column}' does not exist. Please run 'Remove Irrelevant Elements' first.")

        # Step 3: Lemmatization
        if st.button("Perform Lemmatization"):
            def spacy_lemmatizer(text):
                doc = nlp(text)
                return [token.lemma_ for token in doc if not token.is_space]

            cleaned_column = "cleaned_text"
            if cleaned_column in session_state.data.columns:
                session_state.data["lemmatized_tokens"] = session_state.data[cleaned_column].apply(spacy_lemmatizer)

                st.success("Lemmatization completed successfully!")
                st.write(session_state.data[["tokens", "lemmatized_tokens"]].head())
            else:
                st.error(f"'{cleaned_column}' does not exist. Please run 'Remove Irrelevant Elements' first.")

        # Step 4: Keyword Filtering
        keywords = st.text_input("Enter keywords for filtering (comma-separated)", "nuclear, energy, renewable")
        if st.button("Apply Keyword Filtering"):
            lemmatized_column = "lemmatized_tokens"
            filtered_column = "filtered_data"

            if lemmatized_column in session_state.data.columns:
                keyword_list = [kw.strip().lower() for kw in keywords.split(",")]

                def filter_keywords(tokens):
                    # Check if any keyword exists in the list of tokens
                    return any(kw in tokens for kw in keyword_list)

                session_state.data[filtered_column] = session_state.data[lemmatized_column].apply(filter_keywords)
                session_state.data = session_state.data[session_state.data[filtered_column]]

                st.success("Keyword filtering applied successfully!")
                st.write(session_state.data.head())
            else:
                st.error(f"'{lemmatized_column}' does not exist. Please run 'Perform Lemmatization' first.")

        # Save or Undo Changes
        st.subheader("Actions")
        if st.button("Save Preprocessed Data"):
            session_state.preprocessing_backup = session_state.data.copy()
            st.success("Preprocessed data saved successfully!")

        if st.button("Undo Preprocessing Changes"):
            if 'preprocessing_backup' in session_state:
                session_state.data = session_state.preprocessing_backup.copy()
                st.success("Changes reverted to the last saved state!")
                st.write(session_state.data.head())
            else:
                st.error("No backup found. Please save preprocessed data before undoing changes.")
    else:
        st.error("No data loaded. Please upload a dataset to begin preprocessing.")



def get_column_by_name(session_state, possible_names):
    """Find a column in the dataset based on a list of possible names (case-insensitive)."""
    for col in session_state.data.columns:
        if col.lower() in [name.lower() for name in possible_names]:
            return col
    return None


analyzer = SentimentIntensityAnalyzer()

def analyze_sentiment(text):
    scores = analyzer.polarity_scores(text)
    if scores['compound'] >= 0.05:
        return 'positive'
    elif scores['compound'] <= -0.05:
        return 'negative'
    else:
        return 'neutral'


def render_analysis(session_state, analyzer, security):
    # Ensure data is loaded
    if 'data' not in session_state or session_state.data is None:
        st.error("Preprocessed data not found. Please ensure preprocessing is completed and saved.")
        return

    st.title("Data Analysis Methods and Visualization")

    # Detect relevant columns, including 'cleaned_text'
    text_column = get_column_by_name(session_state, ["text", "Text", "cleaned_text"])
    sentiment_column = get_column_by_name(session_state, ["sentiment", "Sentiment"])

    if text_column is None:
        st.error("No text column found for sentiment analysis. Please ensure preprocessing has been completed.")
        return

# If sentiment column is missing, perform sentiment analysis on the selected text column
    if sentiment_column is None:
        with st.spinner("Performing sentiment analysis..."):
            session_state.data['sentiment'] = session_state.data[text_column].apply(analyze_sentiment)
            sentiment_column = 'sentiment'
            st.success("Sentiment analysis completed. 'sentiment' column added to dataset.")

    # Let the user select analysis options from a dropdown
    analysis_type = st.selectbox(
        "Select a Data Analysis Method",
        [
            "Select Method",
            "Sentiment Distribution Analysis",
            "Word Frequency Analysis",
            "Temporal Sentiment Trend Analysis",
            "Keyword-Based Filtering and Analysis",
            "Clustering and Thematic Grouping",

        ]
    )

    # Perform Sentiment Distribution Analysis
    if analysis_type == "Sentiment Distribution Analysis":
        with st.spinner("Generating sentiment distribution..."):
            sentiment_counts = session_state.data[sentiment_column].value_counts()
            st.bar_chart(sentiment_counts)
        st.success("Sentiment distribution visualized!")

    # Perform Word Frequency Analysis
    if analysis_type == "Word Frequency Analysis":
      if text_column is not None:
        with st.spinner("Generating word cloud..."):
            sentiment = st.radio("Select Sentiment", ["positive", "negative", "neutral"])
            words = session_state.data.loc[session_state.data[sentiment_column] == sentiment, text_column]

            if not words.empty:
                wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(words))
                st.image(wordcloud.to_array(), use_container_width=True)
                st.success("Word cloud generated!")
            else:
                st.warning(f"No data available for sentiment '{sentiment}'.")
    else:
        st.warning("No text column found in the dataset.")

    # Generate Word Clouds
    if analysis_type == "Word Cloud Generation":
        if text_column is not None:
            with st.spinner("Generating word cloud..."):
                sentiment = st.radio("Select Sentiment", ["positive", "negative", "neutral"])
                words = session_state.data.loc[session_state.data[sentiment_column] == sentiment, text_column]
                wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(words))
                st.image(wordcloud.to_array(), use_container_width=True)
                st.success("Word cloud generated!")
        else:
            st.warning("No text column found in the dataset.")

    # Perform Temporal Sentiment Trend Analysis
    if analysis_type == "Temporal Sentiment Trend Analysis":
        if 'date' in session_state.data.columns:
            with st.spinner("Analyzing sentiment trends over time..."):
                session_state.data['date'] = pd.to_datetime(session_state.data['date'])
                trend_data = session_state.data.groupby([session_state.data['date'].dt.date, sentiment_column]).size().unstack()
                st.line_chart(trend_data)
            st.success("Sentiment trends analyzed!")
        else:
            st.warning("No 'date' column found in the dataset.")

    # Perform Keyword-Based Filtering and Analysis
    if analysis_type == "Keyword-Based Filtering and Analysis":
        if text_column is not None:
            with st.spinner("Filtering text by keywords..."):
                keywords = st.text_input("Enter keywords separated by commas (e.g., nuclear, renewable)").split(',')
                filtered_data = session_state.data[session_state.data[text_column].str.contains('|'.join(keywords), case=False, na=False)]
                filtered_sentiment_counts = filtered_data[sentiment_column].value_counts()
                st.bar_chart(filtered_sentiment_counts)
            st.success("Keyword-based filtering completed!")
        else:
            st.warning("No text column found in the dataset.")

    # Perform Clustering and Thematic Grouping
    if analysis_type == "Clustering and Thematic Grouping":
        if text_column is not None:
            with st.spinner("Performing clustering and thematic grouping..."):
                vectorizer = TfidfVectorizer(max_features=500)
                tfidf_matrix = vectorizer.fit_transform(session_state.data[text_column])
                kmeans = KMeans(n_clusters=3, random_state=42)
                clusters = kmeans.fit_predict(tfidf_matrix)
                session_state.data['cluster'] = clusters
                st.bar_chart(session_state.data['cluster'].value_counts())
            st.success("Clustering and thematic grouping completed!")
        else:
            st.warning("No text column found in the dataset.")

    # Preview analyzed data
    st.write("Preview of analyzed data:")
    st.dataframe(session_state.data.head())


def render_model_training(session_state, analyzer, security):
    st.title("Model Training and Evaluation")

    # Ensure data is loaded
    if 'data' not in session_state or session_state.data is None:
        st.error("No data loaded. Please load and preprocess data first.")
        return

    # Check for the presence of the 'cleaned_text' and 'sentiment' columns
    text_column = get_column_by_name(session_state, ["cleaned_text"])
    sentiment_column = get_column_by_name(session_state, ["sentiment"])

    if text_column is None or sentiment_column is None:
        st.error("Cleaned text or sentiment column not found. Please preprocess data first.")
        return

    # Encode sentiment labels
    label_mapping = {"positive": 1, "negative": 0, "neutral": 2}
    session_state.data['label'] = session_state.data[sentiment_column].map(label_mapping)

    # Drop rows with missing labels
    session_state.data = session_state.data.dropna(subset=['label'])

    # Split data into features and labels
    X = session_state.data[text_column]
    y = session_state.data['label']

    # Split into training and testing sets
    test_size = st.slider("Select test set size (%)", 10, 50, 20)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size / 100, random_state=42)

    # Handle class imbalance using SMOTE
    smote = SMOTE(random_state=42)
    vectorizer = TfidfVectorizer(max_features=500)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_tfidf, y_train)

    # Model selection
    model_option = st.selectbox("Select a Model", ["SVM", "Naive Bayes", "LSTM"])

    if model_option == "SVM":
        st.write("### SVM Hyperparameter Tuning")
        C = st.slider("Select regularization parameter (C)", 0.01, 10.0, 1.0)
        kernel = st.selectbox("Select kernel type", ["linear", "rbf", "poly", "sigmoid"])

        with st.spinner("Training SVM model..."):
            model = SVC(C=C, kernel=kernel, probability=True)
            model.fit(X_train_resampled, y_train_resampled)
            y_pred = model.predict(X_test_tfidf)
            y_prob = model.predict_proba(X_test_tfidf)

    elif model_option == "Naive Bayes":
        st.write("### Naive Bayes Hyperparameter Tuning")
        alpha = st.slider("Select smoothing parameter (alpha)", 0.01, 2.0, 1.0)

        with st.spinner("Training Naive Bayes model..."):
            model = MultinomialNB(alpha=alpha)
            model.fit(X_train_resampled, y_train_resampled)
            y_pred = model.predict(X_test_tfidf)
            y_prob = model.predict_proba(X_test_tfidf)

    elif model_option == "LSTM":
        st.write("### LSTM Model Configuration")

        max_words = 500
        embedding_dim = 128
        lstm_units = st.slider("Select number of LSTM units", 32, 256, 64)
        epochs = st.slider("Select number of epochs", 1, 20, 5)

        # Tokenize the text data
        tokenizer = Tokenizer(num_words=max_words)
        tokenizer.fit_on_texts(X_train)

        # Apply tokenization on the training and test sets
        X_train_seq = pad_sequences(tokenizer.texts_to_sequences(X_train), maxlen=50)
        X_test_seq = pad_sequences(tokenizer.texts_to_sequences(X_test), maxlen=50)

        # Handle class imbalance with SMOTE on the sequences
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train_seq, y_train)

        # Convert labels to one-hot encoding for LSTM
        y_train_resampled_one_hot = to_categorical(y_train_resampled, num_classes=3)
        y_test_one_hot = to_categorical(y_test, num_classes=3)

        with st.spinner("Training LSTM model..."):
            model = Sequential([
                Embedding(max_words, embedding_dim, input_length=50),
                SpatialDropout1D(0.2),
                LSTM(lstm_units, dropout=0.2, recurrent_dropout=0.2),
                Dense(3, activation='softmax')
            ])

            model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
            history = model.fit(X_train_resampled, y_train_resampled_one_hot, epochs=epochs, validation_data=(X_test_seq, y_test_one_hot))

            y_pred = model.predict(X_test_seq).argmax(axis=1)
            y_prob = model.predict(X_test_seq)

    # Model evaluation
    st.write("### Model Evaluation Metrics")

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    confusion = confusion_matrix(y_test, y_pred)

    st.write(f"**Accuracy**: {accuracy:.2f}")
    st.write(f"**Precision**: {precision:.2f}")
    st.write(f"**Recall**: {recall:.2f}")
    st.write(f"**F1 Score**: {f1:.2f}")

    # Display confusion matrix
    st.write("### Confusion Matrix")
    fig, ax = plt.subplots()
    sns.heatmap(confusion, annot=True, fmt='d', cmap='Blues', xticklabels=label_mapping.keys(), yticklabels=label_mapping.keys())
    st.pyplot(fig)

    # ROC-AUC Score (for SVM and Naive Bayes only)
    if model_option != "LSTM":
        roc_auc = roc_auc_score(y_test, y_prob, multi_class='ovr')
        st.write(f"**ROC-AUC Score**: {roc_auc:.2f}")

    # Save the model
    if st.button("Save Model"):
        model_filename = f"{model_option.lower()}_sentiment_model.pkl"
        if model_option == "LSTM":
            model.save(model_filename)
        else:
            joblib.dump(model, model_filename)
        st.success(f"Model saved as {model_filename}")

    # Store model metrics with model name
    if 'all_model_metrics' not in session_state:
        session_state['all_model_metrics'] = {}

    session_state['all_model_metrics'][model_option] = {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
}

def render_reports(session_state, analyzer, security):
    # Confirm cleaned data availability
    if 'data' not in session_state or session_state.data is None:
        st.warning("Please load and preprocess data first.")
        return

    st.title("Generate Reports")

    # Dropdown for selecting the report to generate
    report_type = st.selectbox(
        "Select a Report to Generate",
        [
            "Select Report",
            "Sentiment Analysis Summary Report",
            "Top Keywords and Themes Report",
            "Model Evaluation Report"
        ]
    )

    # Generate and display the selected report
    if st.button("Generate Report"):
        with st.spinner(f"Generating {report_type}..."):
            report_data = {}
            report_content = None

            # Sentiment Analysis Summary Report
            if report_type == "Sentiment Analysis Summary Report":
                if 'sentiment' in session_state.data.columns:
                    # Count sentiment distribution
                    sentiment_counts = session_state.data['sentiment'].value_counts()
                    st.write("### Sentiment Distribution")
                    st.bar_chart(sentiment_counts)

                    # Display the table with sentiment counts
                    st.write("### Sentiment Distribution Table")
                    sentiment_table = pd.DataFrame({
                        'Sentiment': sentiment_counts.index,
                        'Count': sentiment_counts.values
                    })
                    st.table(sentiment_table)

                    report_data = {
                        'timestamp': datetime.now().isoformat(),
                        'sentiment_distribution': sentiment_counts.to_dict()
                    }
                else:
                    st.warning("Sentiment data not found. Please run sentiment analysis first.")

            # Top Keywords and Themes Report
            elif report_type == "Top Keywords and Themes Report":
                if 'cleaned_text' in session_state.data.columns and 'sentiment' in session_state.data.columns:
                    st.write("### Top Keywords for Each Sentiment")

                    # Function to get top keywords for a specific sentiment
                    def get_top_keywords(data, sentiment, num_keywords=20):
                        sentiment_data = data[data['sentiment'] == sentiment]
                        words = ' '.join(sentiment_data['cleaned_text']).split()
                        word_freq = pd.Series(words).value_counts().head(num_keywords)
                        return word_freq

                    # Display top keywords for positive sentiment
                    st.write("#### Positive Sentiment Keywords")
                    positive_keywords = get_top_keywords(session_state.data, 'positive')
                    positive_keywords = positive_keywords.sort_values(ascending=False)
                    st.bar_chart(positive_keywords)
                    st.table(positive_keywords.reset_index().rename(columns={'index': 'Word', 0: 'Count'}))

                    # Display top keywords for negative sentiment
                    st.write("#### Negative Sentiment Keywords")
                    negative_keywords = get_top_keywords(session_state.data, 'negative')
                    negative_keywords = negative_keywords.sort_values(ascending=False)
                    st.bar_chart(negative_keywords)
                    st.table(negative_keywords.reset_index().rename(columns={'index': 'Word', 0: 'Count'}))

                    # Display top keywords for neutral sentiment
                    st.write("#### Neutral Sentiment Keywords")
                    neutral_keywords = get_top_keywords(session_state.data, 'neutral')
                    neutral_keywords = neutral_keywords.sort_values(ascending=False)
                    st.bar_chart(neutral_keywords)
                    st.table(neutral_keywords.reset_index().rename(columns={'index': 'Word', 0: 'Count'}))

                    # Prepare report data
                    report_data = {
                        'timestamp': datetime.now().isoformat(),
                        'top_positive_keywords': positive_keywords.to_dict(),
                        'top_negative_keywords': negative_keywords.to_dict(),
                        'top_neutral_keywords': neutral_keywords.to_dict()
                    }

                else:
                    st.warning("Cleaned text or sentiment data not found. Please preprocess and analyze sentiment first.")

            # Model Evaluation Report
            elif report_type == "Model Evaluation Report":
                if 'all_model_metrics' in session_state and session_state['all_model_metrics']:
                    st.write("### Model Evaluation Metrics Comparison")

                    # Convert the stored metrics to a DataFrame
                    metrics_df = pd.DataFrame.from_dict(session_state['all_model_metrics'], orient='index')

                    # Add a 'Model' column for display purposes
                    metrics_df.insert(0, 'Model', metrics_df.index)

                    # Display the metrics table
                    st.table(metrics_df)

                    report_data = {
                        'timestamp': datetime.now().isoformat(),
                        'model_metrics': metrics_df.to_dict(orient='records')
                    }
                else:
                    st.warning("No model evaluation metrics found. Please train and evaluate models first.")

            else:
                st.warning("Please select a valid report type.")
                return

            st.success(f"{report_type} generated successfully!")

            # Allow the user to specify a custom file name
            custom_file_name = st.text_input("Enter file name (without extension)", "sentiment_analysis_report")

            # Select the file format
            file_format = st.radio("Select file format to save the report:", ["PDF", "Word", "Excel"])

            # Save the file when the button is clicked
            if st.button("Save Report"):
                # Create the full file name with the selected format
                filename = f"{custom_file_name}.{file_format.lower()}"

                # Generate the report based on the selected format
                if file_format == "PDF":
                    pdf = FPDF()
                    pdf.add_page()
                    pdf.set_font("Arial", size=12)
                    for key, value in report_data.items():
                        pdf.multi_cell(0, 10, f"{key}: {value}")
                    pdf.output(filename)

                elif file_format == "Word":
                    doc = docx.Document()
                    doc.add_heading(report_type, level=1)
                    for key, value in report_data.items():
                        doc.add_paragraph(f"{key}: {value}")
                    doc.save(filename)

                elif file_format == "Excel":
                    df = pd.DataFrame.from_dict(report_data, orient='index')
                    df.to_excel(filename, index=True)

                st.success(f"Report saved as {filename}")

                # Provide a download button for the saved file
                with open(filename, "rb") as file:
                    st.download_button(
                        label=f"Download {file_format} Report",
                        data=file,
                        file_name=filename,
                        mime=f"application/{file_format.lower()}"
                    )

def render_help(session_state, analyzer, security):
    st.title("Help & Documentation")

    st.markdown("""
    ### Analysis Methods
    - **Sentiment Analysis**: Uses VADER and TextBlob for sentiment scoring
    - **Text Statistics**: Basic text metrics and patterns

    ### Data Format Requirements
    - CSV files with a 'text' column
    - Plain text files (one entry per line)
    - Direct text input

    ### Security Features
    - File validation and size limits
    - Input sanitization
    - Basic authentication
    """)

if __name__ == "__main__":
    main()
