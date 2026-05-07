# app.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from wordcloud import WordCloud
import numpy as np

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Hidden Theme Detection",
    layout="wide"
)

# -----------------------------
# TITLE
# -----------------------------
st.title("🧠 Detect Hidden Themes in Complaint Logs")
st.markdown(
    "Analyze customer complaints using NLP and detect hidden issue categories automatically."
)

# -----------------------------
# SIDEBAR
# -----------------------------
st.sidebar.header("⚙ Settings")

num_clusters = st.sidebar.slider(
    "Select Number of Clusters",
    min_value=2,
    max_value=6,
    value=4
)

# -----------------------------
# SAMPLE DATA
# -----------------------------
sample_complaints = """
The delivery was delayed by five days.
Refund process is very slow and frustrating.
Customer support did not answer my calls.
The app crashes every time I open it.
Payment failed but money was deducted.
Product quality is poor and damaged.
Website loading speed is very slow.
The package arrived broken.
Support team resolved my issue quickly.
The mobile app interface is confusing.
Delivery tracking information is incorrect.
Received wrong item in the package.
Refund has not been credited yet.
Technical support is unhelpful.
The checkout page freezes frequently.
Product stopped working after one week.
Customer care executive was rude.
The order arrived earlier than expected.
Payment gateway has many bugs.
The application keeps logging me out.
"""

# -----------------------------
# INPUT SECTION
# -----------------------------
st.subheader("📥 Input Complaint Logs")

input_method = st.radio(
    "Choose Input Method",
    ["Use Sample Data", "Paste Complaints", "Upload CSV"]
)

reviews = []

# SAMPLE
if input_method == "Use Sample Data":
    complaint_text = st.text_area(
        "Complaint Logs",
        sample_complaints,
        height=300
    )

    reviews = [
        line.strip()
        for line in complaint_text.split("\n")
        if len(line.strip()) > 5
    ]

# PASTE
elif input_method == "Paste Complaints":
    complaint_text = st.text_area(
        "Paste Complaints (One complaint per line)",
        height=300
    )

    reviews = [
        line.strip()
        for line in complaint_text.split("\n")
        if len(line.strip()) > 5
    ]

# CSV
else:
    uploaded_file = st.file_uploader(
        "Upload CSV File",
        type=["csv"]
    )

    if uploaded_file is not None:
        df_upload = pd.read_csv(uploaded_file)

        st.write("### Uploaded Data")
        st.dataframe(df_upload.head())

        column_name = st.selectbox(
            "Select Complaint Column",
            df_upload.columns
        )

        reviews = (
            df_upload[column_name]
            .dropna()
            .astype(str)
            .tolist()
        )

# -----------------------------
# ANALYZE BUTTON
# -----------------------------
if st.button("🚀 Analyze Complaints"):

    if len(reviews) < num_clusters:
        st.error("Not enough complaints for clustering.")
        st.stop()

    # -----------------------------
    # TF-IDF
    # -----------------------------
    vectorizer = TfidfVectorizer(
        stop_words='english',
        max_features=1000
    )

    X = vectorizer.fit_transform(reviews)

    # -----------------------------
    # K-MEANS
    # -----------------------------
    kmeans = KMeans(
        n_clusters=num_clusters,
        random_state=42,
        n_init=10
    )

    clusters = kmeans.fit_predict(X)

    # -----------------------------
    # DATAFRAME
    # -----------------------------
    df = pd.DataFrame({
        "Complaint": reviews,
        "Cluster": clusters
    })

    # -----------------------------
    # CLUSTER LABELS
    # -----------------------------
    labels = {
        0: "Delivery Issues",
        1: "Refund Problems",
        2: "Technical Errors",
        3: "Customer Support",
        4: "Payment Issues",
        5: "Product Quality"
    }

    df["Theme"] = df["Cluster"].map(labels)

    # -----------------------------
    # RESULTS
    # -----------------------------
    st.subheader("📊 Clustered Complaints")

    st.dataframe(df)

    # -----------------------------
    # CLUSTER COUNTS
    # -----------------------------
    cluster_counts = df["Theme"].value_counts()

    col1, col2 = st.columns(2)

    # BAR CHART
    with col1:
        st.subheader("📈 Theme Distribution")

        fig, ax = plt.subplots(figsize=(6, 4))

        cluster_counts.plot(
            kind='bar',
            ax=ax
        )

        plt.xticks(rotation=45)
        st.pyplot(fig)

    # PIE CHART
    with col2:
        st.subheader("🥧 Complaint Share")

        fig2, ax2 = plt.subplots(figsize=(6, 6))

        ax2.pie(
            cluster_counts,
            labels=cluster_counts.index,
            autopct='%1.1f%%'
        )

        st.pyplot(fig2)

    # -----------------------------
    # PCA VISUALIZATION
    # -----------------------------
    st.subheader("🧩 Complaint Clusters Visualization")

    pca = PCA(n_components=2)

    reduced = pca.fit_transform(X.toarray())

    pca_df = pd.DataFrame({
        "x": reduced[:, 0],
        "y": reduced[:, 1],
        "Cluster": df["Theme"]
    })

    fig3, ax3 = plt.subplots(figsize=(8, 6))

    sns.scatterplot(
        data=pca_df,
        x="x",
        y="y",
        hue="Cluster",
        s=100
    )

    st.pyplot(fig3)

    # -----------------------------
    # WORD CLOUD
    # -----------------------------
    st.subheader("☁ Common Complaint Words")

    all_text = " ".join(reviews)

    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color='white',
        stopwords=ENGLISH_STOP_WORDS
    ).generate(all_text)

    fig4, ax4 = plt.subplots(figsize=(10, 5))

    ax4.imshow(wordcloud, interpolation='bilinear')
    ax4.axis('off')

    st.pyplot(fig4)

    # -----------------------------
    # TOP KEYWORDS
    # -----------------------------
    st.subheader("🔑 Top Keywords")

    terms = vectorizer.get_feature_names_out()

    sums = X.sum(axis=0)

    word_freq = [
        (word, sums[0, idx])
        for word, idx in vectorizer.vocabulary_.items()
    ]

    word_freq = sorted(
        word_freq,
        key=lambda x: x[1],
        reverse=True
    )[:10]

    keywords_df = pd.DataFrame(
        word_freq,
        columns=["Word", "Score"]
    )

    fig5, ax5 = plt.subplots(figsize=(8, 5))

    sns.barplot(
        data=keywords_df,
        x="Score",
        y="Word",
        ax=ax5
    )

    st.pyplot(fig5)

    # -----------------------------
    # SUMMARY
    # -----------------------------
    st.subheader("📝 Insights Summary")

    most_common_theme = cluster_counts.idxmax()

    st.success(
        f"The most common complaint category is '{most_common_theme}'."
    )

    st.write(
        f"Total Complaints Analyzed: {len(reviews)}"
    )

    st.write(
        f"Number of Hidden Themes Detected: {num_clusters}"
    )

# -----------------------------
# FOOTER
# -----------------------------
st.markdown("---")
st.markdown(
    "Built using Streamlit, NLP, TF-IDF, and K-Means Clustering."
)
