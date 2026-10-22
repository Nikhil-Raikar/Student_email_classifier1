# Vector-Based Email Classifier

A Streamlit demo that embeds labeled historical emails with `all-MiniLM-L6-v2`, stores
the vectors in an in-memory Chroma collection, and predicts a new email's category using
the label of its closest match.

## Files

- `app.py` — Streamlit interface and vector search logic
- `requirements.txt` — Python dependencies
- `email_classification_dataset_final.csv` — add your sanitized/demo dataset here; it must include `subject`, `body`, and `label`

## Run locally

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

## Deploy

Push the project to GitHub, then deploy `app.py` at https://share.streamlit.io/.

## Privacy

Do not commit real confidential emails, personal information, credentials, or API keys.
Use a sanitized or synthetic dataset for a public demo. This app is a demonstration, not a
production email security or routing system.

## Important implementation note

The app builds the Chroma search collection in memory when it starts. This keeps the
starter project simple and avoids committing a generated vector database folder. The
first startup can take a little while because the embedding model must download and the
email vectors must be created.
