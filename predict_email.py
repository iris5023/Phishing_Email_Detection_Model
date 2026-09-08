"""
predict_email.py
------------------
Load the saved trained pipeline and classify one or more new emails.

Usage:
    python3 predict_email.py
    (edit the `emails` list below, or import classify_emails() elsewhere)
"""

import joblib
from features import EmailFeatureExtractor  # noqa: F401 (needed for unpickling)


def load_model(path="phishing_model.joblib"):
    return joblib.load(path)


def classify_emails(emails, model=None):
    if model is None:
        model = load_model()
    preds = model.predict(emails)
    probs = model.predict_proba(emails) if hasattr(model, "predict_proba") else None
    results = []
    for i, (email, pred) in enumerate(zip(emails, preds)):
        label = "Phishing" if pred == 1 else "Safe"
        confidence = float(probs[i][pred]) if probs is not None else None
        results.append({"email": email, "label": label, "confidence": confidence})
    return results


if __name__ == "__main__":
    emails = [
        "Subject: Your account has been limited\nFrom: support@paypa1-alerts.com\n\n"
        "We noticed unusual sign-in activity. Verify your identity now at "
        "http://paypal-secure-check.tk/login or your account will be suspended within 24 hours!",

        "Subject: Team lunch on Friday\nFrom: hr@yourcompany.com\n\n"
        "Hi all, just a reminder that we're having a team lunch this Friday at 1 PM. "
        "Let us know if you have any dietary restrictions.",
    ]

    model = load_model()
    for r in classify_emails(emails, model=model):
        conf = f"{r['confidence']:.1%}" if r["confidence"] is not None else "n/a"
        print(f"[{r['label']} | confidence {conf}]")
        print(r["email"].split("\n\n")[0])
        print("-" * 50)
