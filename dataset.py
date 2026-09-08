"""
dataset.py
----------
Generates a labeled, synthetic-but-realistic dataset of phishing and
legitimate emails for training/testing the classifier.

If you have a REAL dataset (e.g. a CSV with columns 'text' and 'label'
where label is 'phishing'/'safe' or 1/0), you can skip this generator
entirely -- see load_real_dataset() below and the README.
"""

import random
import pandas as pd

random.seed(42)

# ---------------------------------------------------------------------------
# Building blocks used to assemble varied, realistic-sounding emails.
# Mixing and matching templates + fillers gives the TF-IDF vectorizer and the
# engineered features enough diversity to learn real signal instead of
# memorizing a handful of fixed sentences.
# ---------------------------------------------------------------------------

PHISHING_SENDERS = [
    "security@paypa1-support.com", "no-reply@amaz0n-billing.com",
    "alerts@bankofamerica-secure.net", "support@apple-id-verify.com",
    "admin@microsoft-office365-update.com", "helpdesk@netfliix-billing.com",
    "service@irs-taxrefund-gov.com", "info@linkedin-security-check.com",
]

LEGIT_SENDERS = [
    "notifications@github.com", "no-reply@accounts.google.com",
    "hr@yourcompany.com", "newsletter@medium.com",
    "receipts@amazon.com", "team@slack.com",
    "billing@spotify.com", "updates@linkedin.com",
]

PHISHING_SUBJECTS = [
    "Urgent: Your Account Will Be Suspended",
    "Action Required: Verify Your Identity Now",
    "Security Alert - Unusual Login Detected",
    "Your Payment Failed - Update Billing Info Immediately",
    "Congratulations! You've Won a Prize - Claim Now",
    "Final Notice: Account Suspension Pending",
    "Confirm Your Password to Avoid Deactivation",
    "You Have a Pending Refund - Claim Within 24 Hours",
]

LEGIT_SUBJECTS = [
    "Your weekly summary is ready",
    "Receipt for your recent order",
    "New sign-in to your account",
    "Meeting reminder: Team sync at 3 PM",
    "Your subscription renews next month",
    "Here's what you missed this week",
    "Invoice #4821 from your workspace",
    "Welcome to the team!",
]

PHISHING_URLS = [
    "http://192.168.1.44/verify-login", "http://paypa1-support.com/reset",
    "http://bit.ly/3xR9zQa", "http://secure-appleid-check.tk/login",
    "http://amaz0n-account-update.info/confirm",
    "http://192.0.2.55/secure/update.php",
    "http://bankofamerica-alert.xyz/verify",
]

LEGIT_URLS = [
    "https://github.com/settings/notifications",
    "https://accounts.google.com/security",
    "https://www.amazon.com/orders",
    "https://app.slack.com/client",
    "https://www.spotify.com/account",
    "https://www.linkedin.com/feed",
]

URGENT_PHRASES = [
    "act now", "verify your account immediately", "your account has been suspended",
    "click here to confirm your identity", "unusual activity detected on your account",
    "your payment could not be processed", "avoid permanent suspension",
    "update your information within 24 hours", "failure to respond will result in account closure",
    "you have been selected to receive a reward", "confirm your password to continue",
    "limited time offer, respond immediately",
]

NORMAL_PHRASES = [
    "here is a summary of your recent activity", "thanks for being a valued customer",
    "let us know if you have any questions", "your order has shipped",
    "the meeting has been scheduled for tomorrow", "please find the attached invoice",
    "you can update your preferences anytime", "we appreciate your continued support",
    "here are some updates from your team", "your next payment is scheduled for",
]

SIGNOFFS_PHISH = [
    "The Security Team", "Account Services", "Customer Verification Department",
    "Billing Support", "Fraud Prevention Team",
]
SIGNOFFS_LEGIT = [
    "The Team", "Customer Support", "Billing Department", "HR Team", "Notifications",
]


def _make_phishing_email():
    subject = random.choice(PHISHING_SUBJECTS)
    sender = random.choice(PHISHING_SENDERS)
    url = random.choice(PHISHING_URLS)
    n_phrases = random.randint(2, 4)
    body_phrases = random.sample(URGENT_PHRASES, n_phrases)
    exclaim = "!" * random.randint(1, 3)
    body = (
        f"Dear Customer{exclaim} We detected {random.choice(['a problem', 'suspicious activity', 'an issue'])} "
        f"with your account. {' '.join(p.capitalize() + '.' for p in body_phrases)} "
        f"Please visit {url} to resolve this immediately{exclaim} "
        f"If you do not act within {random.choice(['24 hours', '12 hours', '48 hours'])}, "
        f"your account will be permanently locked. "
        f"Regards, {random.choice(SIGNOFFS_PHISH)}"
    )
    text = f"Subject: {subject}\nFrom: {sender}\n\n{body}"
    return text, url, sender


def _make_legit_email():
    subject = random.choice(LEGIT_SUBJECTS)
    sender = random.choice(LEGIT_SENDERS)
    url = random.choice(LEGIT_URLS)
    n_phrases = random.randint(1, 3)
    body_phrases = random.sample(NORMAL_PHRASES, n_phrases)
    body = (
        f"Hi there, {' '.join(p.capitalize() + '.' for p in body_phrases)} "
        f"You can view more details here: {url}. "
        f"Thanks, {random.choice(SIGNOFFS_LEGIT)}"
    )
    text = f"Subject: {subject}\nFrom: {sender}\n\n{body}"
    return text, url, sender


def _make_hard_legit_email():
    """A legitimate email that still uses one urgency-adjacent word, so the
    model can't rely purely on keyword-spotting."""
    subject = random.choice(["Please confirm your appointment", "Reminder: update your password",
                              "Security tip: review your recent sign-ins"])
    sender = random.choice(LEGIT_SENDERS)
    url = random.choice(LEGIT_URLS)
    body = (
        f"Hi, this is a friendly reminder to confirm your details when you get a chance. "
        f"No immediate action is required, but you can review everything at {url}. "
        f"Thanks, {random.choice(SIGNOFFS_LEGIT)}"
    )
    return f"Subject: {subject}\nFrom: {sender}\n\n{body}"


def _make_hard_phishing_email():
    """A phishing email with no obvious IP/shortener URL and calmer tone,
    so the model can't rely purely on the URL heuristics either."""
    subject = random.choice(["Password expiring soon", "Please review your recent order"])
    sender = random.choice(PHISHING_SENDERS)
    body = (
        f"Hello, our records show your password will expire soon. "
        f"To keep uninterrupted access, please sign in and confirm your details at "
        f"http://account-verification-center.com/login. "
        f"Regards, {random.choice(SIGNOFFS_PHISH)}"
    )
    return f"Subject: {subject}\nFrom: {sender}\n\n{body}"


def generate_dataset(n_phishing=350, n_legit=350, hard_fraction=0.12):
    rows = []
    n_hard_p = int(n_phishing * hard_fraction)
    n_hard_l = int(n_legit * hard_fraction)
    for _ in range(n_phishing - n_hard_p):
        text, url, sender = _make_phishing_email()
        rows.append({"text": text, "label": "phishing"})
    for _ in range(n_hard_p):
        rows.append({"text": _make_hard_phishing_email(), "label": "phishing"})
    for _ in range(n_legit - n_hard_l):
        text, url, sender = _make_legit_email()
        rows.append({"text": text, "label": "safe"})
    for _ in range(n_hard_l):
        rows.append({"text": _make_hard_legit_email(), "label": "safe"})
    df = pd.DataFrame(rows).sample(frac=1, random_state=42).reset_index(drop=True)
    return df


def load_real_dataset(csv_path, text_col="text", label_col="label"):
    """
    Load a real-world dataset instead of the synthetic one.
    Expects a CSV with a text column and a label column.
    Labels can be strings ('phishing'/'safe', 'spam'/'ham', etc.) or 0/1 --
    they'll be normalized to 'phishing' / 'safe'.
    """
    df = pd.read_csv(csv_path)
    df = df.rename(columns={text_col: "text", label_col: "label"})[["text", "label"]]

    def _norm(v):
        s = str(v).strip().lower()
        if s in ("1", "phishing", "spam", "malicious", "phish", "true", "yes"):
            return "phishing"
        return "safe"

    df["label"] = df["label"].apply(_norm)
    return df.dropna(subset=["text"]).reset_index(drop=True)


if __name__ == "__main__":
    df = generate_dataset()
    print(df["label"].value_counts())
    print(df.iloc[0]["text"])
