import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Load dataset
df = pd.read_csv("Dataset.csv")

# Keep required columns only 
df = df[['review_text', 'rating']]

# Drop missing reviews
df.dropna(inplace=True)

# Remap rating 2 → 1 (Poor)
df['rating'] = df['rating'].replace(2, 1)

# Features & target
X = df['review_text']
y = df['rating']

# Vectorizer
vectorizer = TfidfVectorizer(
    stop_words='english',
    max_features=5000
)

X_vec = vectorizer.fit_transform(X)

# Model
model = LogisticRegression(max_iter=2000)
model.fit(X_vec, y)

# Save model
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("✅ Model trained using dataset and saved successfully")
