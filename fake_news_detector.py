# Fake News Detection - Machine Learning Project

import pandas as pd
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Download stopwords if not already downloaded
nltk.download('stopwords')

# Step 1: Load the data
fake = pd.read_csv("Fake.csv")
real = pd.read_csv("True.csv")

# Add labels: 0 = fake, 1 = real
fake['label'] = 0
real['label'] = 1

# Combine and shuffle the data
data = pd.concat([fake, real])
data = data[['text', 'label']]
data = data.sample(frac=1).reset_index(drop=True)

# Step 2: Text Cleaning
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def clean_text(text):
    text = re.sub('[^a-zA-Z]', ' ', text)  # Remove punctuation/numbers
    text = text.lower().split()  # Lowercase and split
    text = [stemmer.stem(word) for word in text if word not in stop_words]
    return ' '.join(text)

data['text'] = data['text'].apply(clean_text)

# Step 3: Vectorize using TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(data['text']).toarray()
y = data['label']

# Step 4: Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Train the Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Step 6: Evaluate the model
y_pred = model.predict(X_test)
print("✅ Accuracy:", accuracy_score(y_test, y_pred))
print("\n📊 Classification Report:\n", classification_report(y_test, y_pred))

# Step 7: Make prediction on custom input
def predict_news(news_text):
    cleaned = clean_text(news_text)
    vectorized = vectorizer.transform([cleaned]).toarray()
    prediction = model.predict(vectorized)
    return "REAL" if prediction[0] == 1 else "FAKE"

# Test the prediction function
sample_news = "Breaking: Government announces new AI law"
print("\n📰 Sample News Prediction:", predict_news(sample_news))
