# ✨ Toxic Comment Detector

## Overview
A Streamlit web app that detects harmful and toxic online comments using a pre-trained multi-label machine-learning model. Enter a comment and the app returns a probability for each toxicity category, plus a clear safe/toxic verdict, all in a pastel dashboard UI.

## Features
- Six separate toxicity categories, read directly from the model: Toxic, Severe Toxic, Obscene, Threat, Insult, Identity Hate
- Per-category probability bars and category cards
- Summary dashboard (comment stats, safety status, toxicity score, categories detected)
- Friendly handling of empty input, invalid input, a missing model file and prediction errors (details go to the server logs)
- Custom pastel theme, responsive layout, sidebar navigation

## Technologies Used
- Python
- Streamlit
- Scikit-learn (Logistic Regression classifiers)
- NLP / TF-IDF (word 1–2-grams and character 2–5-grams)
- Machine Learning (one binary classifier per label)

## How It Works
`toxic_debiased.joblib` is a dictionary containing two fitted `TfidfVectorizer`s (`word_vec`, `char_vec`), six fitted `LogisticRegression` models (`models`) and the label list (`labels`). For each comment the app:
1. Vectorizes the text with both TF-IDF vectorizers and concatenates the features (100,000 total).
2. Calls `predict_proba` on each label's classifier.
3. Marks a category as **Detected** when its probability is ≥ 50% (the default decision threshold).

The model is only loaded and used for inference; it is never retrained or modified.

## Project Structure
```
toxic-comment-detector/
├── app.py                  # Streamlit app (UI + inference)
├── toxic_debiased.joblib   # Pre-trained model (~8.5 MB)
├── requirements.txt
├── README.md
├── .gitignore
└── .streamlit/config.toml  # Light pastel theme settings
```

## Installation
```bash
git clone https://github.com/<your-username>/toxic-comment-detector.git
cd toxic-comment-detector
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```
`scikit-learn` is pinned to **1.6.1**, the version the model was saved with.

## Running Locally
```bash
streamlit run app.py
```
Then open http://localhost:8501.

## Deployment (Streamlit Community Cloud)
1. Push this project to your GitHub repository.
2. Go to https://share.streamlit.io and sign in with GitHub.
3. Click **Create app** → choose the `toxic-comment-detector` repo, branch `main`, main file `app.py`.
4. Open **Advanced settings** and select a Python version of 3.11 or 3.12.
5. Click **Deploy**. Streamlit installs `requirements.txt` automatically.

## Disclaimer
Predictions are automated and can be wrong. Use them as guidance, not as a final moderation decision.
