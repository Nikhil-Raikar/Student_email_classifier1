import streamlit as st
import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer

st.set_page_config(page_title="Email Classifier", page_icon="📧", layout="wide")

st.title("📧 Vector-Based Email Classifier")
st.write(
    "Enter an email subject and body. The app searches labeled historical emails "
    "and predicts the category using the closest match."
)

DATA_PATH = "email_classification_dataset_final.csv"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
TOP_K = 3


@st.cache_resource(show_spinner="Loading the embedding model and building the email search index...")
def load_resources(csv_path: str):
    """Load the CSV, create embeddings, and build an in-memory Chroma collection."""
    df = pd.read_csv(csv_path)

    required = {"subject", "body", "label"}
    missing = required - set(df.columns.str.lower())
    if missing:
        raise ValueError(
            "CSV must contain these columns: subject, body, label. "
            f"Missing columns: {sorted(missing)}"
        )

    # Normalize column names so Subject/Body/Label also work.
    df.columns = [str(c).strip().lower() for c in df.columns]
    df = df.dropna(subset=["label"]).copy()
    df["subject"] = df["subject"].fillna("").astype(str)
    df["body"] = df["body"].fillna("").astype(str)
    df["label"] = df["label"].astype(str).str.strip()
    df = df[df["label"] != ""].copy()

    # Match the project's original text representation.
    df["combined_text"] = "Subject: " + df["subject"] + "\n\nBody: " + df["body"]
    df = df.drop_duplicates(subset=["subject", "body", "label"]).reset_index(drop=True)

    if df.empty:
        raise ValueError("The CSV has no usable labeled email rows.")

    model = SentenceTransformer(EMBEDDING_MODEL)
    texts = df["combined_text"].tolist()
    embeddings = model.encode(
        texts,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    client = chromadb.Client()
    collection = client.create_collection(
        name="email_examples",
        metadata={"hnsw:space": "cosine"},
    )

    # Chroma IDs must be unique strings.
    ids = [f"email_{i}" for i in range(len(df))]
    collection.add(
        ids=ids,
        documents=texts,
        embeddings=embeddings.tolist(),
        metadatas=[
            {
                "label": row["label"],
                "subject": row["subject"][:500],
                "body": row["body"][:1500],
            }
            for _, row in df.iterrows()
        ],
    )
    return model, collection, len(df), sorted(df["label"].unique().tolist())


try:
    model, collection, reference_count, labels = load_resources(DATA_PATH)
except FileNotFoundError:
    st.error(
        f"Couldn't find `{DATA_PATH}`. Upload your labeled CSV to the same GitHub "
        "repository folder as `app.py`."
    )
    st.stop()
except Exception as exc:
    st.error(f"Couldn't load the dataset or build the search index: {exc}")
    st.stop()

with st.sidebar:
    st.header("About this demo")
    st.metric("Reference emails indexed", reference_count)
    st.metric("Available categories", len(labels))
    st.caption(f"Embedding model: {EMBEDDING_MODEL}")
    st.caption("The index is rebuilt when the app starts or its cached resources are cleared.")

subject = st.text_input("Email subject", placeholder="Example: Your interview is scheduled")
body = st.text_area(
    "Email body",
    height=220,
    placeholder="Paste the email body here...",
)

if st.button("Classify email", type="primary", use_container_width=True):
    if not subject.strip() and not body.strip():
        st.warning("Please enter an email subject or body first.")
    else:
        query_text = f"Subject: {subject.strip()}\n\nBody: {body.strip()}"
        query_embedding = model.encode(
            [query_text],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )[0]

        results = collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=min(TOP_K, reference_count),
            include=["documents", "metadatas", "distances"],
        )

        matches = []
        for doc, metadata, distance in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            matches.append(
                {
                    "label": metadata.get("label", "Unknown"),
                    "subject": metadata.get("subject", ""),
                    "body": metadata.get("body", ""),
                    "distance": float(distance),
                    "document": doc,
                }
            )

        # Baseline behavior: use the label of the single closest historical email.
        prediction = matches[0]["label"]

        st.subheader("Prediction")
        st.success(f"Predicted category: **{prediction}**")
        st.caption(
            "This is a similarity-based prediction, not a calibrated probability. "
            "A smaller cosine distance means a closer match."
        )
        st.metric("Closest-match cosine distance", f"{matches[0]['distance']:.4f}")

        st.subheader("Top similar historical emails")
        for i, match in enumerate(matches, start=1):
            with st.expander(
                f"Match {i} · {match['label']} · distance {match['distance']:.4f}",
                expanded=(i == 1),
            ):
                st.markdown(f"**Subject:** {match['subject'] or '(empty subject)'}")
                st.markdown("**Body excerpt:**")
                st.write(match["body"][:1000] or "(empty body)")
                st.caption("Use these examples to understand why the search returned this category.")

st.divider()
st.caption(
    "Demo only: predictions can be wrong. Do not paste confidential, personal, or "
    "sensitive emails into a public demo."
)
