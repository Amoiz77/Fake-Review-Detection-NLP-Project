import joblib
import nltk
import pandas as pd
import re
import scipy.sparse as sp
import streamlit as st
from scipy.sparse import hstack

# Page setup
st.set_page_config(page_title="Fake Review Detection Studio", page_icon="🛡️", layout="centered")

# NLTK setup for gibberish filtering
nltk.download("words", quiet=True)
from nltk.corpus import words

english_vocab = set(words.words())


# -----------------------------------------------------------------------------
# 1. Load Model Artifacts
# -----------------------------------------------------------------------------
@st.cache_resource
def load_backend():
    model = joblib.load("model.pkl")
    tfidf_word = joblib.load("tfidf_word.pkl")
    tfidf_char = joblib.load("tfidf_char.pkl")
    return model, tfidf_word, tfidf_char


try:
    model, tfidf_word, tfidf_char = load_backend()
except Exception:
    st.error("❌ Missing model artifacts! Please run 'train.py' first.")
    st.stop()


# -----------------------------------------------------------------------------
# 2. Helpers
# -----------------------------------------------------------------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z\s]", "", text)
    return text.strip()


def check_mid_sentence(text):
    t = str(text).strip()
    if not t: return 0
    t = t.rstrip("\"')]")
    if not t: return 0
    last = t[-1]
    if last in ".!?" or last.isdigit(): return 0
    return 1


def is_meaningful_text(text, threshold=0.3):
    cleaned_words = clean_text(text).split()
    if not cleaned_words: return False
    valid_words = sum(1 for word in cleaned_words if word in english_vocab)
    return (valid_words / len(cleaned_words)) >= threshold


def is_promotional_spam(text):
    """Rule-based guardrail for blatant promotional spam."""
    text_lower = text.lower()
    spam_triggers = ["buy now", "click here", "win prize", "limited time offer", "free offer"]
    has_trigger = any(trigger in text_lower for trigger in spam_triggers)

    heavy_caps = (sum(1 for c in text if c.isupper()) / max(len(text), 1)) > 0.4 and len(text) > 10
    heavy_exclamation = text.count("!") >= 3

    return has_trigger or (heavy_caps and heavy_exclamation)


def extract_features(df):
    raw = df["text_"].astype(str)
    clean = df["clean_text"]

    features = pd.DataFrame()
    features["text_length"] = clean.apply(len)
    features["word_count"] = clean.apply(lambda x: len(x.split()))
    features["exclamation_count"] = raw.apply(lambda x: x.count("!"))
    features["upper_ratio"] = raw.apply(lambda x: sum(1 for c in x if c.isupper()) / max(len(x), 1))

    spam_keywords = ["buy now", "click here", "limited time", "win prize", "offer", "free", "spam"]
    features["spam_keyword_count"] = clean.apply(lambda x: sum(1 for kw in spam_keywords if kw in x))
    features["is_excessive_caps"] = (features["upper_ratio"] > 0.4).astype(int)
    features["is_excessive_exclamation"] = (features["exclamation_count"] >= 3).astype(int)

    repeated_phrases = [
        "i love the look and feel of this pillow", "the only problem is that its not really a",
        "the only reason i gave it 4 stars", "i also love that its removable",
        "we love this blanket", "i will keep my", "very pretty",
    ]
    for phrase in repeated_phrases:
        col = "has_" + phrase[:20].replace(" ", "_")
        features[col] = clean.apply(lambda x: int(phrase in x))

    positive_words = ["love", "great", "perfect", "excellent", "amazing", "best", "awesome"]
    features["positive_word_count"] = clean.apply(lambda x: sum(1 for w in positive_words if w in x.split()))

    features["rating"] = pd.to_numeric(df["rating"], errors="coerce").fillna(3.0)
    features["sentiment_mismatch"] = ((features["rating"] <= 2) & (features["positive_word_count"] >= 2)).astype(int)
    features["ends_mid_sentence"] = raw.apply(check_mid_sentence)
    features["boilerplate_phrase"] = clean.apply(lambda x: int("the only problem is" in x or "i will keep my" in x))

    return features


# -----------------------------------------------------------------------------
# 3. User Interface
# -----------------------------------------------------------------------------
st.title("🛡️ Fake Review Detection System")
st.markdown("Analyze product reviews to check whether they are **Genuine (OR)** or **Fake (CG)**.")
st.divider()

review_input = st.text_area("Enter Review Text:",
                            value="I love the look and feel of this pillow. The only problem is that it's not really a",
                            height=120)
rating_input = st.slider("Assigned Star Rating:", 1.0, 5.0, 5.0, 1.0)

if st.button("Analyze Review", type="primary", use_container_width=True):
    if not review_input.strip():
        st.warning("Please enter a review to analyze.")
    elif not is_meaningful_text(review_input):
        st.error("⚠️ **Invalid Input:** The text entered appears to be random gibberish.")
    else:
        st.divider()
        st.subheader("Analysis Output")
        col1, col2 = st.columns(2)

        # 1. Check Hard-Rule Spam First
        if is_promotional_spam(review_input):
            col1.error("### ⚠️ Fake (Promotional Spam)")
            col2.metric("Prediction Confidence", "99.00% (Rule-Based)")

        # 2. Otherwise, Fallback to Model
        else:
            df_single = pd.DataFrame({
                "text_": [review_input],
                "clean_text": [clean_text(review_input)],
                "rating": [rating_input],
            })

            feats = extract_features(df_single)
            tw = tfidf_word.transform(df_single["clean_text"])
            tc = tfidf_char.transform(df_single["clean_text"])
            tf = sp.csr_matrix(feats.values)
            X = hstack([tw, tc, tf])

            pred = model.predict(X)[0]
            probs = model.predict_proba(X)[0]

            # FRIEND'S FIX #1 APPLIED: pred == 0 is Fake (CG), pred == 1 is Genuine (OR)
            if pred == 0:
                col1.error("### ⚠️ Fake (CG)")
                confidence = probs[0] * 100
            else:
                col1.success("### ✅ Genuine (OR)")
                confidence = probs[1] * 100

            col2.metric("Prediction Confidence", f"{confidence:.2f}% (Model)")

