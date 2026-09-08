"""
phishing_detector.py
---------------------
End-to-end phishing email classifier.

Pipeline:
  raw email text
      |--> TfidfVectorizer            -> word-level signal
      |--> EmailFeatureExtractor       -> URL / keyword / structural signal
      v
  FeatureUnion -> combined feature matrix
      v
  Classifier (Random Forest, compared against Logistic Regression & Naive Bayes)
      v
  Accuracy, Precision/Recall/F1, Confusion Matrix

Run:
    python3 phishing_detector.py

Outputs:
    confusion_matrix.png   - heatmap of the best model's confusion matrix
    model_comparison.png   - bar chart comparing model accuracies
    phishing_model.joblib  - the trained pipeline, ready to reuse
"""

import joblib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import FunctionTransformer, MinMaxScaler
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    ConfusionMatrixDisplay,
)

from dataset import load_real_dataset
from features import EmailFeatureExtractor


def build_feature_pipeline():
    """TF-IDF on raw text + engineered numeric features, combined."""
    tfidf = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=2000, ngram_range=(1, 2), stop_words="english")),
    ])
    engineered = Pipeline([
        ("extract", EmailFeatureExtractor()),
        ("scale", MinMaxScaler()),
    ])
    return FeatureUnion([("tfidf", tfidf), ("engineered", engineered)])


def build_models():
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "Multinomial Naive Bayes": MultinomialNB(),
        "Random Forest": RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced"),
    }


def main():
    print("=" * 60)
    print("PHISHING EMAIL DETECTION - Training Pipeline")
    print("=" * 60)

    # 1. Load data -----------------------------------------------------
    df = load_real_dataset("Phishing_Email.csv", text_col="Email Text", label_col="Email Type")
    print(f"\nDataset: {len(df)} emails "
          f"({(df.label == 'phishing').sum()} phishing / {(df.label == 'safe').sum()} safe)")

    X = df["text"].values
    y = (df["label"] == "phishing").astype(int).values  # 1 = phishing, 0 = safe

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    print(f"Train: {len(X_train)}  |  Test: {len(X_test)}")

    # 2. Try a few models, pick the best on held-out test accuracy -----
    results = {}
    fitted_pipelines = {}

    for name, clf in build_models().items():
        pipe = Pipeline([
            ("features", build_feature_pipeline()),
            ("clf", clf),
        ])
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        acc = accuracy_score(y_test, preds)
        results[name] = acc
        fitted_pipelines[name] = pipe
        print(f"\n--- {name} ---")
        print(f"Accuracy: {acc:.4f}")
        print(classification_report(y_test, preds, target_names=["safe", "phishing"]))

    best_name = max(results, key=results.get)
    best_pipe = fitted_pipelines[best_name]
    print(f"\nBest model: {best_name}  (accuracy = {results[best_name]:.4f})")

    # 3. Confusion matrix for the best model ----------------------------
    y_pred = best_pipe.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix (rows=actual, cols=predicted) [safe, phishing]:")
    print(cm)

    fig, ax = plt.subplots(figsize=(5.5, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Safe", "Phishing"])
    disp.plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
    ax.set_title(f"Confusion Matrix - {best_name}\nAccuracy: {results[best_name]:.2%}")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150)
    plt.close()

    # 4. Model comparison chart ------------------------------------------
    fig, ax = plt.subplots(figsize=(6, 4))
    names = list(results.keys())
    accs = [results[n] for n in names]
    bars = ax.bar(names, accs, color=["#4C72B0", "#DD8452", "#55A868"])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Accuracy")
    ax.set_title("Model Comparison")
    for b, a in zip(bars, accs):
        ax.text(b.get_x() + b.get_width() / 2, a + 0.02, f"{a:.2%}", ha="center")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig("model_comparison.png", dpi=150)
    plt.close()

    # 5. Save the best pipeline -------------------------------------------
    joblib.dump(best_pipe, "phishing_model.joblib")
    print("\nSaved trained pipeline -> phishing_model.joblib")
    print("Saved charts -> confusion_matrix.png, model_comparison.png")

    # 6. Demo: classify a couple of brand-new example emails --------------
    demo_emails = [
        "Subject: Verify Your Account Now\nFrom: security@paypa1-alerts.com\n\n"
        "Dear user, we noticed unusual activity. Click http://192.168.5.2/verify "
        "immediately to avoid suspension of your account!",

        "Subject: Your invoice is ready\nFrom: billing@spotify.com\n\n"
        "Hi, your monthly invoice is attached. You can view it anytime at "
        "https://www.spotify.com/account. Thanks for being a subscriber!",
    ]
    demo_preds = best_pipe.predict(demo_emails)
    demo_proba = best_pipe.predict_proba(demo_emails) if hasattr(best_pipe, "predict_proba") else None

    print("\n--- Demo predictions on new emails ---")
    for i, (email, pred) in enumerate(zip(demo_emails, demo_preds)):
        label = "PHISHING" if pred == 1 else "SAFE"
        conf = f" (confidence: {demo_proba[i][pred]:.2%})" if demo_proba is not None else ""
        print(f"\nEmail {i+1} -> {label}{conf}")
        print(email.split('\n\n')[0])

    return best_pipe, results


if __name__ == "__main__":
    main()
