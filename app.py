import sys
import os
import subprocess
from streamlit.runtime.scriptrunner import get_script_run_ctx

# 🛡️ Auto-launch Streamlit if the user runs this file directly with Python!
if not get_script_run_ctx():
    print("🚀 Automatically starting Streamlit server...")
    subprocess.run([sys.executable, "-m", "streamlit", "run", os.path.abspath(__file__)])
    sys.exit()

import streamlit as st
import pickle
import os
import json
from project import clean_text  # We can reuse the cleaning function you already wrote!

# Set up the look and feel of the web page
st.set_page_config(page_title="Sentiment Analysis", page_icon="🎬", layout="centered")

# Add a title and description
st.title("🎬 Movie Review Sentiment Analyzer")
st.write("Welcome to the Machine Learning Sentiment Analysis App! Type a movie review below, and the AI will determine if it's Positive or Negative.")

model_file = 'sentiment_model.pkl'
vectorizer_file = 'tfidf_vectorizer.pkl'

# Load the model (using st.cache_resource so it only loads once, making the app super fast)
@st.cache_resource
def load_model():
    if os.path.exists(model_file) and os.path.exists(vectorizer_file):
        with open(model_file, 'rb') as f:
            model = pickle.load(f)
        with open(vectorizer_file, 'rb') as f:
            vectorizer = pickle.load(f)
        return model, vectorizer
    return None, None

model, vectorizer = load_model()

if model is None or vectorizer is None:
    st.error("⚠️ Model files not found! Please run `project.py` first to train the model.")
else:
    # File to store our community dictionary
    DICT_FILE = 'community_dictionary.json'

    def load_dict():
        if os.path.exists(DICT_FILE):
            with open(DICT_FILE, 'r') as f:
                return json.load(f)
        return {}

    def save_dict(d):
        with open(DICT_FILE, 'w') as f:
            json.dump(d, f)

    # Create tabs for the UI
    tab1, tab2, tab3 = st.tabs(["🎬 Analyze", "🧠 Teach the AI", "🤖 Movie Chatbot"])

    with tab1:
        # Create a text box for user input
        user_input = st.text_area("📝 Enter your movie review here:", 
                                  placeholder="e.g., This movie was absolutely fantastic! I loved every minute of it.", 
                                  height=150)
        
        # Create an "Analyze" button
        if st.button("Analyze Sentiment"):
            if user_input.strip() == "":
                st.warning("Please enter some text to analyze.")
            else:
                with st.spinner("Analyzing text..."):
                    cleaned_text = clean_text(user_input)
                    vectorized_text = vectorizer.transform([cleaned_text])
                    
                    # Get base probability (Confidence Score) from the model
                    probs = model.predict_proba(vectorized_text)[0]
                    pos_prob = probs[1] # Probability of being Positive
                    
                    # Apply Hybrid Logic: Check the Community Dictionary
                    comm_dict = load_dict()
                    modifier = 0.0
                    applied_words = []
                    
                    for word in cleaned_text.split():
                        if word in comm_dict:
                            votes = comm_dict[word]
                            total = votes['pos'] + votes['neg']
                            # Threshold: Needs 5 votes and 75% agreement
                            if total >= 5:
                                if votes['pos'] / total >= 0.75:
                                    modifier += 0.15 # Boost positive score by 15%
                                    applied_words.append((word, "Positive"))
                                elif votes['neg'] / total >= 0.75:
                                    modifier -= 0.15 # Boost negative score by 15%
                                    applied_words.append((word, "Negative"))
                    
                    # Calculate final prediction
                    final_pos_prob = min(max(pos_prob + modifier, 0.0), 1.0)
                    prediction = 1 if final_pos_prob >= 0.5 else 0
                    
                    # Display the results with confidence percentages
                    if prediction == 1:
                        st.success(f"### 🟢 Positive Sentiment! ({final_pos_prob*100:.1f}% Confident)")
                        st.balloons()
                    else:
                        st.error(f"### 🔴 Negative Sentiment! ({(1-final_pos_prob)*100:.1f}% Confident)")
                        
                    if applied_words:
                        st.info(f"💡 The community dictionary adjusted this score based on user-taught words: {', '.join([w[0] for w in applied_words])}")

    with tab2:
        st.header("Community Dictionary")
        st.write("Teach the AI modern slang or words it gets wrong! Words need at least **5 votes** and **75% agreement** to affect predictions.")
        
        new_word = st.text_input("Enter a word to vote on (e.g., 'mid', 'fire', 'cap'):").strip().lower()
        
        col1, col2 = st.columns(2)
        if col1.button("Vote Positive 👍"):
            if new_word:
                d = load_dict()
                if new_word not in d: d[new_word] = {"pos": 0, "neg": 0}
                d[new_word]["pos"] += 1
                save_dict(d)
                st.success(f"Voted '{new_word}' as Positive!")
        
        if col2.button("Vote Negative 👎"):
            if new_word:
                d = load_dict()
                if new_word not in d: d[new_word] = {"pos": 0, "neg": 0}
                d[new_word]["neg"] += 1
                save_dict(d)
                st.success(f"Voted '{new_word}' as Negative!")
                
        st.divider()
        st.write("### Current Dictionary Status")
        d = load_dict()
        if d:
            for word, votes in d.items():
                total = votes['pos'] + votes['neg']
                status = "⏳ Pending (Needs more votes)"
                if total >= 5:
                    if votes['pos'] / total >= 0.75: 
                        status = "✅ Accepted (Positive)"
                    elif votes['neg'] / total >= 0.75: 
                        status = "✅ Accepted (Negative)"
                    else: 
                        status = "❌ Disputed (Mixed votes)"
                st.write(f"**{word}** - 👍 {votes['pos']} | 👎 {votes['neg']} | Status: {status}")
        else:
            st.write("The dictionary is empty. Be the first to add a word!")

    with tab3:
        st.header("🤖 Movie Recommendation Bot")
        st.write("Chat with me! Ask for a movie genre and I'll give you a recommendation.")
        
        # 1. Initialize chat history in Streamlit's memory (session_state)
        if "messages" not in st.session_state:
            st.session_state.messages = [
                {"role": "assistant", "content": "Hi! I'm your movie bot 🍿. What kind of movie are you in the mood for? (Try: action, comedy, horror, sci-fi, romance)"}
            ]

        # 2. Display existing chat messages from history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # 3. Accept user input
        if prompt := st.chat_input("Ask for a recommendation..."):
            # Display user message in chat message container
            with st.chat_message("user"):
                st.markdown(prompt)
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})

            # 4. Generate the Bot's response (Rule-based keyword matching)
            user_text = prompt.lower()
            if "action" in user_text:
                bot_reply = "🔥 For action, you can't go wrong with **Mad Max: Fury Road** or **The Dark Knight**!"
            elif "comedy" in user_text or "funny" in user_text:
                bot_reply = "😂 Need a laugh? I highly recommend **Superbad** or **The Hangover**."
            elif "horror" in user_text or "scary" in user_text:
                bot_reply = "👻 If you want to be scared, check out **The Conjuring** or **Get Out**!"
            elif "sci-fi" in user_text or "space" in user_text:
                bot_reply = "👽 For sci-fi fans, **Interstellar** and **The Matrix** are absolute masterpieces."
            elif "romance" in user_text or "love" in user_text:
                bot_reply = "❤️ For a great romance, watch **The Notebook** or **La La Land**."
            else:
                bot_reply = "🤔 I'm a simple bot right now! Try asking me for genres like **action, comedy, horror, sci-fi, or romance**."

            # Display assistant response in chat message container
            with st.chat_message("assistant"):
                st.markdown(bot_reply)
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": bot_reply})