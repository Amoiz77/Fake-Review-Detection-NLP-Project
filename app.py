import joblib
import pandas as pd
import re
import scipy.sparse as sp
import streamlit as st
from scipy.sparse import hstack

# Set up page styling
st.set_page_config(
    page_title="Fake Review Detection Studio", page_icon="🛡️", layout="wide"
)

# -----------------------------------------------------------------------------
# 1. Load Model & Vectorizers (Cached for Performance)
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
        "❌ Critical Error: Missing model artifacts (`model.pkl`, `tfidf_word.pkl`, `tfidf_char.pkl`). "
        "Please run your training script first to save these files."
    )
    st.stop()

# -----------------------------------------------------------------------------
# 2. Exact Preprocessing & Feature Extraction Logic
# -----------------------------------------------------------------------------


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z\s]", "", text)
    return text.strip()


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

    features["ends_mid_sentence"] = raw.apply(
        lambda x: int(str(x).strip()[-1] not in ".!?\"'")
    )

    features["boilerplate_phrase"] = clean.apply(
        lambda x: int("the only problem is" in x or "i will keep my" in x)
    )

    return features


def run_pipeline(reviews, ratings):
    """Executes feature engineering + vectorized transform on raw inputs."""
    df_test = pd.DataFrame(
        {
            "text_": reviews,
            "clean_text": [clean_text(r) for r in reviews],
            "rating": ratings,
        }
    )

    feats = extract_features(df_test)

    tw = tfidf_word.transform(df_test["clean_text"])
    tc = tfidf_char.transform(df_test["clean_text"])
    tf = sp.csr_matrix(feats.values)

    X = hstack([tw, tc, tf])

    preds = model.predict(X)
    probs = model.predict_proba(X)

    return df_test, feats, preds, probs


# -----------------------------------------------------------------------------
# 3. Streamlit Interface Layout
# -----------------------------------------------------------------------------

st.title("🛡️ Fake Review Detection System")
st.markdown(
    "Analyze product reviews using your custom NLP pipeline combining **Word TF-IDF**, **Character n-grams**, and **Behavioral Heuristics**."
)

tab1, tab2 = st.tabs(["Single Review Analysis", "Batch Evaluation (CSV / List)"])

# -----------------------
# TAB 1: Single Review
# -----------------------
with tab1:
    st.subheader("Individual Review Checker")

    col1, col2 = st.columns([3, 1])
    with col1:
        user_review = st.text_area(
            "Review Content:",
            value="I love the look and feel of this pillow. The only problem is that it's not really a",
            height=120,
        )
    with col2:
        user_rating = st.number_input(
            "Product Rating (1-5):",
            min_value=1.0,
            max_value=5.0,
            value=5.0,
            step=1.0,
        )

    if st.button("Run Model Prediction", type="primary"):
        if not user_review.strip():
            st.warning("Please enter text to analyze.")
        else:
            _, feats, preds, probs = run_pipeline([user_review], [user_rating])

            pred = preds[0]
            prob = probs[0]

            st.divider()

            res_col1, res_col2 = st.columns(2)
            with res_col1:
                if pred == 1:
                    st.error("### ⚠️ Result: Fake / Computer Generated (CG)")
                    st.metric(
                        "Prediction Confidence", f"{prob[1] * 100:.2f}%"
                    )
                else:
                    st.success("### ✅ Result: Genuine / Original (OR)")
                    st.metric(
                        "Prediction Confidence", f"{prob[0] * 100:.2f}%"
                    )

            with res_col2:
                st.markdown("**Engineered Signal Matrix**")
                st.json({
                    "Uppercase Ratio": f"{feats['upper_ratio'].iloc[0]:.2f}",
                    "Sentiment Mismatch Flag": int(
                        feats["sentiment_mismatch"].iloc[0]
                    ),
                    "Ends Mid-Sentence Flag": int(
                        feats["ends_mid_sentence"].iloc[0]
                    ),
                    "Boilerplate Phrase Flag": int(
                        feats["boilerplate_phrase"].iloc[0]
                    ),
                    "Positive Word Count": int(
                        feats["positive_word_count"].iloc[0]
                    ),
                })

# -----------------------
# TAB 2: Batch Processing
# -----------------------
with tab2:
    st.subheader("Batch Review Tester")
    st.markdown(
        "Upload a `.csv` file containing a `text_` column (and optional `rating` column), or test against pre-built backend samples."
    )

    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

    if uploaded_file is not None:
        batch_df = pd.read_csv(uploaded_file)
        if "text_" not in batch_df.columns:
            st.error("The uploaded CSV must contain a `text_` column.")
        else:
            ratings = (
                batch_df["rating"].tolist()
                if "rating" in batch_df.columns
                else [3.0] * len(batch_df)
            )
            reviews = batch_df["text_"].tolist()

            if st.button("Analyze Uploaded File"):
                _, _, preds, probs = run_pipeline(reviews, ratings)
                batch_df["Predicted Label"] = [
                    "Fake (CG)" if p == 1 else "Genuine (OR)" for p in preds
                ]
                batch_df["Confidence Score"] = [
                    f"{max(pr) * 100:.2f}%" for pr in probs
                ]

                st.dataframe(
                    batch_df[["text_", "Predicted Label", "Confidence Score"]],
                    use_container_width=True,
                )
    else:
        st.info(
            "No CSV uploaded. Running test suite using backend sample list."
        )
        if st.button("Run Test Suite"):
            test_reviews = [
                "Absolutely amazing product, works like a charm",
                "Worst product ever do not buy",
                "Buy now!!! Limited time offer!!!",
                "Good quality and durable",
                "Spam spam spam buy now",
                "I love the look and feel of this pillow. The only problem is that it's not really a",
                "We love this blanket. Very pretty.",
            ]
            default_ratings = [5.0, 1.0, 5.0, 4.0, 1.0, 5.0, 5.0]

            _, _, preds, probs = run_pipeline(test_reviews, default_ratings)

            results_data = []
            for rev, p, pr in zip(test_reviews, preds, probs):
                results_data.append({
                    "Review Preview": rev,
                    "Prediction": "Fake (CG)" if p == 1 else "Genuine (OR)",
                    "Confidence": f"{max(pr) * 100:.2f}%",
                })

            st.table(pd.DataFrame(results_data))
