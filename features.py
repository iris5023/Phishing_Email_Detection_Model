"""
features.py
------------
Hand-engineered features that capture the classic signals used in
phishing-email detection, on top of raw text (which is handled
separately by a TF-IDF vectorizer in model.py).

These are combined into a scikit-learn Pipeline via FeatureUnion, so the
final model sees BOTH the TF-IDF word signal AND these structured
red-flag features.
"""

import re
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

URL_RE = re.compile(r"https?://[^\s]+")
IP_URL_RE = re.compile(r"https?://\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}")

SUSPICIOUS_KEYWORDS = [
    "verify", "urgent", "suspend", "suspended", "immediately", "confirm",
    "click here", "act now", "password", "update your", "limited time",
    "winner", "congratulations", "claim", "account will be", "unusual activity",
    "security alert", "restricted", "locked", "expire", "refund",
]

SUSPICIOUS_TLDS = (".tk", ".xyz", ".info", ".top", ".ru", ".cc")

FREE_SHORTENERS = ("bit.ly", "tinyurl", "goo.gl", "t.co", "is.gd")


def _count_urls(text):
    return len(URL_RE.findall(text))


def _has_ip_url(text):
    return 1 if IP_URL_RE.search(text) else 0


def _has_shortener(text):
    return 1 if any(s in text.lower() for s in FREE_SHORTENERS) else 0


def _has_suspicious_tld(text):
    return 1 if any(tld in text.lower() for tld in SUSPICIOUS_TLDS) else 0


def _keyword_count(text):
    t = text.lower()
    return sum(t.count(k) for k in SUSPICIOUS_KEYWORDS)


def _sender_mismatch(text):
    """Rough heuristic: does the 'From:' domain look like a spoofed brand
    (contains a well-known brand name mixed with extra words/digits)?"""
    m = re.search(r"From:\s*[^\n]*@([^\s\n>]+)", text, re.IGNORECASE)
    if not m:
        return 0
    domain = m.group(1).lower()
    brands = ["paypal", "amazon", "apple", "microsoft", "netflix", "bank", "irs", "linkedin", "google"]
    for b in brands:
        if b in domain and domain.split(".")[0] not in (b,):
            # brand name present but not as the clean root domain -> suspicious
            if not domain.startswith(b + "."):
                return 1
    return 0


def extract_row_features(text):
    return {
        "num_urls": _count_urls(text),
        "has_ip_url": _has_ip_url(text),
        "has_shortener": _has_shortener(text),
        "has_suspicious_tld": _has_suspicious_tld(text),
        "keyword_count": _keyword_count(text),
        "num_exclamations": text.count("!"),
        "num_capital_words": sum(1 for w in text.split() if w.isupper() and len(w) > 2),
        "text_length": len(text),
        "sender_mismatch": _sender_mismatch(text),
    }


FEATURE_NAMES = list(extract_row_features("dummy https://a.com").keys())


class EmailFeatureExtractor(BaseEstimator, TransformerMixin):
    """Scikit-learn-compatible transformer: raw email text -> numeric feature matrix."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        rows = [extract_row_features(t) for t in X]
        arr = np.array([[r[name] for name in FEATURE_NAMES] for r in rows], dtype=float)
        return arr

    def get_feature_names_out(self, input_features=None):
        return np.array(FEATURE_NAMES)
