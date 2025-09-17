import streamlit as st
import pickle
import re

# Load pre-trained model & vectorizer
model = pickle.load(open("fake_news_model.pkl", "rb"))
vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))

# Preprocessing function
def preprocess(text):
    text = text.lower()
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text

# Streamlit UI
st.set_page_config(page_title="Fake News Detector", layout="centered")
st.title("📰 Fake News Detection App")
st.markdown("Detect whether the news you entered is **Real** or **Fake** using AI!")

user_input = st.text_area("Enter news content below 👇", height=200)

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter some news text.")
    else:
        clean_text = preprocess(user_input)
        vectorized_input = vectorizer.transform([clean_text])
        prediction = model.predict(vectorized_input)[0]

        if prediction == 1:
            st.success("✅ This looks like *Real News*!")
        else:
            st.error("🚨 This might be *Fake News*!")

st.markdown("---")
st.markdown("Made with ❤️ using Streamlit and Machine Learning")

