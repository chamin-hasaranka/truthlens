

import os
import re
import sys
import string
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
MODEL_DIR = os.path.join(BASE_DIR, "model")

FAKE_CSV = os.path.join(DATASET_DIR, "Fake.csv")
TRUE_CSV = os.path.join(DATASET_DIR, "True.csv")

MODEL_PATH = os.path.join(MODEL_DIR, "fake_news_model.pkl")
VECTORIZER_PATH = os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl")
METRICS_PATH = os.path.join(MODEL_DIR, "metrics.pkl")


def clean_text(text: str) -> str:
    """
    Basic text cleaning used both at training time and at prediction time.
    Keeping this identical in app.py ensures consistent preprocessing.
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)   # remove URLs
    text = re.sub(r"<.*?>", " ", text)                    # remove HTML tags
    text = re.sub(r"\[.*?\]", " ", text)                  # remove bracketed text
    text = re.sub(r"\d+", " ", text)                       # remove numbers
    text = text.translate(str.maketrans("", "", string.punctuation))  # remove punctuation
    text = re.sub(r"\s+", " ", text).strip()               # normalize whitespace
    return text


def load_dataset():
    """Load Fake.csv and True.csv, label them, and merge into a single dataframe."""
    if not os.path.exists(FAKE_CSV) or not os.path.exists(TRUE_CSV):
        print("ERROR: Could not find dataset/Fake.csv and/or dataset/True.csv")
        print("Please place the Fake and Real News dataset CSV files inside the 'dataset' folder.")
        print("You can generate a small sample dataset by running:")
        print("    python dataset/generate_sample_dataset.py")
        sys.exit(1)

    fake_df = pd.read_csv(FAKE_CSV)
    true_df = pd.read_csv(TRUE_CSV)

    # Label: FAKE = 0, REAL = 1
    fake_df["label"] = 0
    true_df["label"] = 1

    df = pd.concat([fake_df, true_df], ignore_index=True)

    # Combine title + text for richer features. Handle missing columns gracefully.
    title_col = df["title"] if "title" in df.columns else ""
    text_col = df["text"] if "text" in df.columns else ""
    df["content"] = (title_col.fillna("") + " " + text_col.fillna("")).str.strip()

    # Drop empty rows
    df = df[df["content"].str.len() > 0].reset_index(drop=True)

    # Shuffle the dataset
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    return df


def main():
    print("=" * 60)
    print("Fake News Detection - Model Training")
    print("=" * 60)

    os.makedirs(MODEL_DIR, exist_ok=True)

    print("\n[1/6] Loading dataset...")
    df = load_dataset()
    print(f"      Loaded {len(df)} total articles "
          f"({(df['label'] == 1).sum()} REAL, {(df['label'] == 0).sum()} FAKE)")

    print("\n[2/6] Cleaning text...")
    df["clean_content"] = df["content"].apply(clean_text)

    print("\n[3/6] Splitting into train/test sets (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(
        df["clean_content"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
    )
    print(f"      Train size: {len(X_train)}  |  Test size: {len(X_test)}")

    print("\n[4/6] Vectorizing text using TF-IDF...")
    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_df=0.7,
        min_df=2,
        ngram_range=(1, 2),
        max_features=50000,
    )
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    print(f"      Vocabulary size: {len(vectorizer.vocabulary_)}")

    print("\n[5/6] Training PassiveAggressiveClassifier...")
    model = PassiveAggressiveClassifier(max_iter=1000, C=0.5, random_state=42)
    model.fit(X_train_tfidf, y_train)

    print("\n[6/6] Evaluating model on test set...")
    y_pred = model.predict(X_test_tfidf)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    print("\n" + "-" * 60)
    print(f"Accuracy : {accuracy * 100:.2f}%")
    print(f"Precision: {precision * 100:.2f}%")
    print(f"Recall   : {recall * 100:.2f}%")
    print(f"F1 Score : {f1 * 100:.2f}%")
    print("Confusion Matrix:")
    print(cm)
    print("-" * 60)

    # Save model, vectorizer, and metrics
    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    joblib.dump(
        {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "train_size": len(X_train),
            "test_size": len(X_test),
        },
        METRICS_PATH,
    )

    print(f"\nModel saved to:      {MODEL_PATH}")
    print(f"Vectorizer saved to: {VECTORIZER_PATH}")
    print(f"Metrics saved to:    {METRICS_PATH}")
    print("\nTraining complete! You can now run the Flask app with: python app.py")


if __name__ == "__main__":
    main()
