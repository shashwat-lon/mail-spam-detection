# Spam Detection using Python (ML + NLP)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load dataset
# Download spam.csv from https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset
df = pd.read_csv("spam.csv", encoding='latin-1')[['v1', 'v2']]
df.columns = ['label', 'message']  # rename columns

# Convert labels (ham=0, spam=1)
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    df['message'], df['label'], test_size=0.2, random_state=42
)

# Convert text to features
cv = CountVectorizer(stop_words='english')   # bag of words
X_train_counts = cv.fit_transform(X_train)

# Use TF-IDF (better than raw counts)
tfidf = TfidfTransformer()
X_train_tfidf = tfidf.fit_transform(X_train_counts)

# Train model (Naive Bayes is great for text classification)
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# Transform test set
X_test_counts = cv.transform(X_test)
X_test_tfidf = tfidf.transform(X_test_counts)

# Predictions
y_pred = model.predict(X_test_tfidf)

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
