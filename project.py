import os
from dotenv import load_dotenv
import pickle
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Download required NLTK datasets (only needs to be run once)
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

# ==========================================
# Step 1: Load the Dataset
# ==========================================
def load_data(filepath):
    print("Loading dataset...")
    # Read the CSV file using Pandas
    df = pd.read_csv(filepath)
    
    # Map the text labels to numerical values (Positive: 1, Negative: 0)
    df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0})
    return df

# ==========================================
# Step 2: Preprocessing
# ==========================================
def clean_text(text):
    # Initialize English stop words
    stop_words = set(stopwords.words('english'))
    
    # Remove HTML tags (e.g., <br />)
    text = re.sub(r'<.*?>', '', text)
    # Remove non-alphabetical characters (punctuation, numbers)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Convert text to lowercase
    text = text.lower()
    
    # Tokenize the text
    tokens = word_tokenize(text)
    
    # Remove stop words
    cleaned_tokens = [word for word in tokens if word not in stop_words]
    
    # Rejoin tokens into a single string
    return ' '.join(cleaned_tokens)

# ==========================================
# Step 5 & 6: Evaluation & Custom Testing
# ==========================================
def evaluate_model(y_test, y_pred):
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nModel Accuracy: {accuracy * 100:.2f}%")
    
    # Generate and plot confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Negative', 'Positive'], 
                yticklabels=['Negative', 'Positive'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Sentiment')
    plt.ylabel('Actual Sentiment')
    plt.show()

def predict_custom_sentence(model, vectorizer, sentence):
    # Preprocess the custom sentence
    cleaned_sentence = clean_text(sentence)
    
    # Transform it using the previously fitted TF-IDF vectorizer
    vectorized_sentence = vectorizer.transform([cleaned_sentence])
    
    # Make a prediction
    prediction = model.predict(vectorized_sentence)
    
    # Output the result
    sentiment = "Positive" if prediction[0] == 1 else "Negative"
    print(f"\nInput: '{sentence}'")
    print(f"Predicted Sentiment: {sentiment}")

# ==========================================
# Main Execution Block
# ==========================================
if __name__ == "__main__":
    # Load environment variables from the .env file
    load_dotenv()

    model_file = 'sentiment_model.pkl'
    vectorizer_file = 'tfidf_vectorizer.pkl'

    # Check if the model and vectorizer are already saved
    if os.path.exists(model_file) and os.path.exists(vectorizer_file):
        print("Loading saved model and vectorizer...")
        with open(model_file, 'rb') as f:
            model = pickle.load(f)
        with open(vectorizer_file, 'rb') as f:
            vectorizer = pickle.load(f)
    else:
        # 1. Load data (Resolves path relative to this script's location)
        dataset_path = os.path.join(os.path.dirname(__file__), 'IMDB Dataset.csv')
        
        # Download dataset automatically if it doesn't exist
        if not os.path.exists(dataset_path):
            print("Dataset not found locally. Downloading from Kaggle...")
            import kaggle
            kaggle.api.authenticate()
            kaggle.api.dataset_download_files('lakshmi25npathi/imdb-dataset-of-50k-movie-reviews', path=os.path.dirname(__file__), unzip=True)
            print("Download and extraction complete!")

        df = load_data(dataset_path)
        
        # 2. Preprocess the text
        print("Cleaning text data... (This might take a couple of minutes for 50k records)")
        df['cleaned_review'] = df['review'].apply(clean_text)
        
        # Split the dataset into training (80%) and testing (20%) sets
        X_train, X_test, y_train, y_test = train_test_split(
            df['cleaned_review'], df['sentiment'], test_size=0.2, random_state=42
        )
        
        # 3. Feature Extraction
        print("Vectorizing text using TF-IDF...")
        vectorizer = TfidfVectorizer(max_features=5000) # Limit to top 5000 words for efficiency
        X_train_tfidf = vectorizer.fit_transform(X_train)
        X_test_tfidf = vectorizer.transform(X_test)
        
        # 4. Model Training
        print("Training Logistic Regression model...")
        model = LogisticRegression()
        model.fit(X_train_tfidf, y_train)
        
        # Predict on the test set and Evaluate
        print("Evaluating model...")
        y_pred = model.predict(X_test_tfidf)
        evaluate_model(y_test, y_pred)

        # Save the model and vectorizer for future use
        print("Saving model and vectorizer...")
        with open(model_file, 'wb') as f:
            pickle.dump(model, f)
        with open(vectorizer_file, 'wb') as f:
            pickle.dump(vectorizer, f)
    
    # 6. Test with custom inputs
    print("\n--- Testing Custom Sentences ---")
    predict_custom_sentence(model, vectorizer, "This movie was absolutely fantastic! I loved every minute of it.")
    predict_custom_sentence(model, vectorizer, "Terrible acting, awful plot. I want my money back.")