import pandas as pd
import pickle
import re
import nltk

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

nltk.download("stopwords")

# Load data
df = pd.read_csv("data/tickets.csv")

stop_words = set(stopwords.words("english"))

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z ]", "", text)
    return " ".join(w for w in text.split() if w not in stop_words)

df["clean_text"] = df["text"].apply(clean_text)

# Vectorization
X = df["clean_text"]
y = df["category"]

vectorizer = TfidfVectorizer()
X_vec = vectorizer.fit_transform(X)

# Train model
model = LogisticRegression()
model.fit(X_vec, y)

# Save model
pickle.dump(model, open("model/classifier.pkl", "wb"))
pickle.dump(vectorizer, open("model/vectorizer.pkl", "wb"))

print("✅ Model trained and saved successfully")
