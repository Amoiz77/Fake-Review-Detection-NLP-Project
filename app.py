import joblib
import nltk
import pandas as pd
import re
import scipy.sparse as sp
import streamlit as st
from scipy.sparse import hstack

# -----------------------------------------------------------------------------
# Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Fake Review Detection Studio", page_icon="🛡️", layout="centered"
)

# NLTK setup: english word list for gibberish filtering + lemmatizer
nltk.download("words", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)
from nltk.corpus import words
from nltk.stem import WordNetLemmatizer

english_vocab = set(words.words())
lemmatizer = WordNetLemmatizer()

# -----------------------------------------------------------------------------
# 1. Load Backend Artifacts
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
    st.error(
        "❌ Missing model files! Please run `python train.py` first to create the necessary model artifacts."
    )
    st.stop()

# -----------------------------------------------------------------------------
# 2. Helper Functions
# -----------------------------------------------------------------------------
def clean_text(text):
    """Must exactly mirror train.py's clean_text, or the TF-IDF vocabulary
    learned at training time won't line up with what's produced here."""
    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-z\s]", "", text)
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(tok) for tok in tokens]
    return " ".join(tokens).strip()


def check_mid_sentence(text):
    t = str(text).strip()
    if not t:
        return 0
    t = t.rstrip("\"')]")
    if not t:
        return 0
    last = t[-1]
    if last in ".!?" or last.isdigit():
        return 0
    return 1


def is_meaningful_text(text, threshold=0.25):
    cleaned_words = clean_text(text).split()
    if not cleaned_words:
        return False
    valid_words = sum(1 for word in cleaned_words if word in english_vocab)
    return (valid_words / len(cleaned_words)) >= threshold


def is_promotional_spam(text):
    text_lower = text.lower()
    spam_triggers = [
        "buy now",
        "click here",
        "win prize",
        "limited time offer",
        "free offer",
    ]
    has_trigger = any(trigger in text_lower for trigger in spam_triggers)
    heavy_caps = (
        sum(1 for c in text if c.isupper()) / max(len(text), 1)
    ) > 0.45 and len(text) > 10
    heavy_exclamation = text.count("!") >= 3
    return has_trigger or (heavy_caps and heavy_exclamation)


def extract_features(df):
    raw = df["text_"].astype(str)
    clean = df["clean_text"]

    features = pd.DataFrame()
    features["text_length"] = clean.apply(len)
    features["word_count"] = clean.apply(lambda x: len(x.split()))
    features["exclamation_count"] = raw.apply(lambda x: x.count("!"))
    features["upper_ratio"] = raw.apply(
        lambda x: sum(1 for c in x if c.isupper()) / max(len(x), 1)
    )

    spam_keywords = [
        "buy now",
        "click here",
        "limited time",
        "win prize",
        "offer",
        "free",
        "spam",
    ]
    features["spam_keyword_count"] = clean.apply(
        lambda x: sum(1 for kw in spam_keywords if kw in x)
    )
    features["is_excessive_caps"] = (features["upper_ratio"] > 0.4).astype(int)
    features["is_excessive_exclamation"] = (
        features["exclamation_count"] >= 3
    ).astype(int)

    repeated_phrases = [
        "i love the look and feel of this pillow",
        "the only problem is that its not really a",
        "the only reason i gave it 4 stars",
        "i also love that its removable",
        "we love this blanket",
        "i will keep my",
        "very pretty",
    ]
    for phrase in repeated_phrases:
        col = "has_" + phrase[:20].replace(" ", "_")
        features[col] = clean.apply(lambda x: int(phrase in x))

    positive_words = [
        "love",
        "great",
        "perfect",
        "excellent",
        "amazing",
        "best",
        "awesome",
    ]
    features["positive_word_count"] = clean.apply(
        lambda x: sum(1 for w in positive_words if w in x.split())
    )

    features["rating"] = pd.to_numeric(df["rating"], errors="coerce").fillna(
        3.0
    )
    features["sentiment_mismatch"] = (
        (features["rating"] <= 2) & (features["positive_word_count"] >= 2)
    ).astype(int)

    features["ends_mid_sentence"] = raw.apply(check_mid_sentence)
    features["boilerplate_phrase"] = clean.apply(
        lambda x: int("the only problem is" in x or "i will keep my" in x)
    )

    return features


def build_feature_matrix(raw_texts, rating):
    """Shared pipeline: raw text(s) + a fixed rating -> full model input."""
    df_batch = pd.DataFrame({"text_": raw_texts})
    df_batch["clean_text"] = df_batch["text_"].apply(clean_text)
    df_batch["rating"] = rating
    feats = extract_features(df_batch)
    tw = tfidf_word.transform(df_batch["clean_text"])
    tc = tfidf_char.transform(df_batch["clean_text"])
    tf = sp.csr_matrix(feats.values)
    return hstack([tw, tc, tf]).tocsr()


# -----------------------------------------------------------------------------
# 3. User Interface
# -----------------------------------------------------------------------------
st.title("🛡️ Fake Review Detection System")
st.markdown(
    "Analyze product reviews to check whether they are **Genuine (OR)** or **Fake (CG)**."
)
st.divider()

review_input = st.text_area(
    "Enter Review Text:",
    value="Good quality and durable, fits well with my living room setup.",
    height=120,
)

rating_input = st.slider("Assigned Star Rating:", 1.0, 5.0, 5.0, 1.0)

with st.expander("⚙️ Advanced Settings"):
    fake_threshold = st.slider(
        "Fake Classification Confidence Threshold:",
        min_value=0.50,
        max_value=0.85,
        value=0.65,
        step=0.05,
        help="Higher values require the model to be more certain before marking a review as Fake.",
    )
    show_xai = st.checkbox("Show why the model made this prediction (XAI)", value=True)

if st.button("Analyze Review", type="primary", use_container_width=True):
    if not review_input.strip():
        st.warning("Please enter a review to analyze.")
    elif not is_meaningful_text(review_input):
        st.error(
            "⚠️ **Invalid Input:** The text entered appears to be random gibberish."
        )
    else:
        st.divider()
        st.subheader("Analysis Output")
        col1, col2 = st.columns(2)

        # 1. Rule-Based Spam Override
        if is_promotional_spam(review_input):
            col1.error("### ⚠️ Fake (Promotional Spam)")
            col2.metric("Prediction Confidence", "99.00% (Rule-Based)")
            st.caption(
                "Flagged by keyword/formatting rules before reaching the ML model, "
                "so no model-based explanation applies here."
            )

        # 2. Model Prediction
        else:
            X = build_feature_matrix([review_input], rating_input)
            probs = model.predict_proba(X)[0]
            fake_prob = probs[0]  # Label 0 = CG / Fake
            genuine_prob = probs[1]  # Label 1 = OR / Genuine

            if fake_prob >= fake_threshold:
                col1.error("### ⚠️ Fake (CG)")
                col2.metric("Prediction Confidence", f"{fake_prob * 100:.2f}%")
            else:
                col1.success("### ✅ Genuine (OR)")
                col2.metric(
                    "Prediction Confidence", f"{genuine_prob * 100:.2f}%"
                )

            # ---------------------------------------------------------------
            # Explainable AI (XAI): LIME token-level explanation for this
            # specific prediction, computed live on the entered review.
            # ---------------------------------------------------------------
            if show_xai:
                st.divider()
                st.subheader("🔍 Why did the model decide this? (XAI)")
                with st.spinner("Computing explanation..."):
                    from lime.lime_text import LimeTextExplainer

                    explainer = LimeTextExplainer(
                        class_names=["CG (fake)", "OR (genuine)"]
                    )

                    def lime_predict_fn(text_list):
                        Xb = build_feature_matrix(list(text_list), rating_input)
                        return model.predict_proba(Xb)

                    exp = explainer.explain_instance(
                        review_input, lime_predict_fn, num_features=8, labels=(0, 1)
                    )
                    contributions = exp.as_list(label=1)

                exp_df = pd.DataFrame(contributions, columns=["token", "weight"])
                exp_df["direction"] = exp_df["weight"].apply(
                    lambda w: "→ Genuine (OR)" if w > 0 else "→ Fake (CG)"
                )
                st.dataframe(
                    exp_df.style.format({"weight": "{:+.4f}"}),
                    use_container_width=True,
                    hide_index=True,
                )
                st.caption(
                    "Positive weight = pushed the prediction toward Genuine (OR). "
                    "Negative weight = pushed the prediction toward Fake (CG)."
                )

with st.expander("📊 Model Transparency: global top predictive words"):
    st.markdown(
        "These are the strongest words the Logistic Regression model learned "
        "overall (not specific to the review above)."
    )
    coefs = model.coef_[0]
    word_vocab = tfidf_word.get_feature_names_out()
    n_word_feats = len(word_vocab)
    word_coefs = coefs[:n_word_feats]

    top_genuine_idx = word_coefs.argsort()[-10:][::-1]
    top_fake_idx = word_coefs.argsort()[:10]

    colA, colB = st.columns(2)
    with colA:
        st.markdown("**Top words → Genuine (OR)**")
        for i in top_genuine_idx:
            st.write(f"`{word_vocab[i]}`  (+{word_coefs[i]:.3f})")
    with colB:
        st.markdown("**Top words → Fake (CG)**")
        for i in top_fake_idx:
            st.write(f"`{word_vocab[i]}`  ({word_coefs[i]:.3f})")
