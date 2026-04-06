# Movie-Review-Sentiment-Analyzer 🎬
An end-to-end NLP pipeline for binary sentiment classification, featuring a unique Human-in-the-Loop (HITL) module to adapt to evolving language and slang.

✨ Key Features
Sentiment Analysis: Classifies IMDB movie reviews as Positive or Negative with ~89% accuracy.

Human-in-the-Loop (HITL): A "Teach the AI" module where users vote on modern colloquialisms. If a term hits consensus (≥5 votes, ≥75% agreement), the system adjusts its logic to handle data drift.

Interactive Web App: Real-time inference dashboard built with Streamlit.

Cinematic Chatbot: A built-in agent for genre-specific movie recommendations.

🛠️ Tech Stack
Core: Python, Scikit-Learn

NLP: NLTK, Regex, TF-IDF Vectorization

ML Model: Logistic Regression

Interface: Streamlit

📊 Pipeline Overview
Preprocessing: Rigorous text cleaning (HTML removal, tokenization, and stop-word filtering).

Feature Extraction: TF-IDF vectorization focused on the top 5,000 contextual features.

Dynamic Refinement: The HITL module allows the community to "patch" the model's understanding of new vocabulary without a full retraining cycle.
