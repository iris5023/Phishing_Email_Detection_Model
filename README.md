Download the dataset from [Kaggle](https://www.kaggle.com/datasets/subhajournal/phishingemails) and place `Phishing_Email.csv` in the project folder before running.
# Phishing Email Detection Model

A Scikit-learn pipeline that classifies emails as **Phishing** or **Safe**
using both the email's text content and engineered red-flag features
(suspicious URLs, urgency keywords, sender spoofing patterns, etc.).

## Files

| File | Purpose |
|---|---|
| `dataset.py` | Generates a labeled synthetic dataset of phishing/legit emails (or loads a real CSV via `load_real_dataset()`) |
| `features.py` | Hand-engineered features: URL count, IP-address URLs, link shorteners, suspicious TLDs, urgency keywords, exclamation marks, sender/domain mismatch, etc. |
| `phishing_detector.py` | Main script — builds the pipeline, trains & compares 3 models, prints accuracy + confusion matrix, saves the best model |
| `predict_email.py` | Loads the saved model and classifies new emails |

## How it works

```
raw email text
   |--> TfidfVectorizer (word/bigram frequencies)   ---\
   |                                                     >-- FeatureUnion --> Classifier --> Phishing / Safe
   |--> EmailFeatureExtractor (URLs, keywords, ...)  ---/
```

Three classifiers are trained and compared on a held-out test split:
**Logistic Regression**, **Multinomial Naive Bayes**, and **Random Forest**.
The best-performing one is saved to `phishing_model.joblib`.

## Run it

```bash
pip install scikit-learn pandas matplotlib joblib
python3 phishing_detector.py
```

This prints:
- Per-model accuracy and a full classification report (precision/recall/F1)
- A confusion matrix (also saved as `confusion_matrix.png`)
- A model comparison bar chart (`model_comparison.png`)
- Demo predictions on two brand-new example emails

Then classify your own emails:

```bash
python3 predict_email.py
```

or from Python:

```python
from predict_email import load_model, classify_emails

model = load_model()
results = classify_emails(["Subject: ... \nFrom: ...\n\nBody text here"], model=model)
print(results)
```

## Using a real dataset instead of the synthetic one

The synthetic generator (`dataset.py`) exists so the project runs
out-of-the-box with no external download. For a stronger, real-world
model, swap it out:

```python
from dataset import load_real_dataset
df = load_real_dataset("your_emails.csv", text_col="text", label_col="label")
```

Any CSV with a text column and a phishing/legit label column works —
common public datasets include the Kaggle "Phishing Email Detection"
and "Spam/Ham" datasets, or the Nazario phishing corpus. Labels are
auto-normalized (`1`/`phishing`/`spam` -> phishing, everything else -> safe).

## Notes on the demo dataset

`dataset.py`'s synthetic emails are template-based, so the model reaches
very high (often ~100%) accuracy on this data — that reflects how
separable the *templates* are, not real-world difficulty. A handful of
"hard" borderline examples are mixed in on purpose. Swap in a real
dataset for a realistic accuracy figure (well-tuned phishing classifiers
on public datasets typically land in the 94–99% range).

## Extending it

- Add features: sender-reply-to mismatch, attachment types, HTML-vs-plaintext ratio, SPF/DKIM pass/fail if available.
- Try `GridSearchCV` to tune `TfidfVectorizer(max_features, ngram_range)` and classifier hyperparameters.
- For deployment, wrap `predict_email.classify_emails()` in a simple Flask/FastAPI endpoint or a browser extension backend.
