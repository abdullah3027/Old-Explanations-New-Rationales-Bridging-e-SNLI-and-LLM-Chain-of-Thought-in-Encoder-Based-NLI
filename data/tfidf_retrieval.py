"""TF-IDF based explanation retrieval for Variant B test-time inference.

Builds a TF-IDF index over training examples and retrieves the most similar
training explanation for each test example based on cosine similarity of
premise + hypothesis text.
"""

import sys
from pathlib import Path

import numpy as np
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from training.config import VariantBConfig
from data.preprocess import load_esnli_train


def build_tfidf_index(config: VariantBConfig = None):
    """Build TF-IDF index over training set premise+hypothesis text.

    Returns:
        vectorizer: Fitted TfidfVectorizer
        train_matrix: Sparse TF-IDF matrix of shape (n_train, n_features)
        train_explanations: List of training explanations
    """
    if config is None:
        config = VariantBConfig()

    print("Loading training data for TF-IDF index...")
    train_df = load_esnli_train(config)

    texts = (train_df["Sentence1"] + " " + train_df["Sentence2"]).tolist()
    train_explanations = train_df["Explanation_1"].tolist()

    print(f"Fitting TF-IDF vectorizer on {len(texts)} examples...")
    vectorizer = TfidfVectorizer(max_features=50000, stop_words="english")
    train_matrix = vectorizer.fit_transform(texts)

    print("TF-IDF index built successfully.")
    return vectorizer, train_matrix, train_explanations


def save_tfidf_index(vectorizer, train_matrix, train_explanations, output_dir: str):
    """Save TF-IDF index to disk for reuse."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    joblib.dump(vectorizer, output_path / "tfidf_vectorizer.joblib")
    joblib.dump(train_matrix, output_path / "tfidf_train_matrix.joblib")
    joblib.dump(train_explanations, output_path / "tfidf_train_explanations.joblib")
    print(f"TF-IDF index saved to {output_path}")


def load_tfidf_index(output_dir: str):
    """Load a previously saved TF-IDF index."""
    output_path = Path(output_dir)
    vectorizer = joblib.load(output_path / "tfidf_vectorizer.joblib")
    train_matrix = joblib.load(output_path / "tfidf_train_matrix.joblib")
    train_explanations = joblib.load(output_path / "tfidf_train_explanations.joblib")
    return vectorizer, train_matrix, train_explanations


def retrieve_explanations(premises, hypotheses, vectorizer, train_matrix, train_explanations, batch_size=1000):
    """Retrieve the most similar training explanation for each test example.

    Processes in batches to manage memory with ~549K training examples.

    Args:
        premises: List of test premise strings
        hypotheses: List of test hypothesis strings
        vectorizer: Fitted TfidfVectorizer
        train_matrix: Sparse TF-IDF matrix from training set
        train_explanations: List of training explanations
        batch_size: Number of test examples to process at once

    Returns:
        List of retrieved explanations (one per test example)
    """
    test_texts = [p + " " + h for p, h in zip(premises, hypotheses)]
    retrieved = []

    for i in range(0, len(test_texts), batch_size):
        batch = test_texts[i : i + batch_size]
        batch_matrix = vectorizer.transform(batch)
        similarities = cosine_similarity(batch_matrix, train_matrix)
        best_indices = np.argmax(similarities, axis=1)
        retrieved.extend([train_explanations[idx] for idx in best_indices])

        if (i // batch_size) % 5 == 0:
            print(f"  Retrieved explanations for {min(i + batch_size, len(test_texts))}/{len(test_texts)} examples")

    return retrieved
