import streamlit as st
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(
    page_title="AI FAQ Chatbot",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 AI FAQ Chatbot")
st.write("Welcome! Ask me anything about AI, Python, programming and technology.")

# Load FAQ data
with open("faqs.json", "r", encoding="utf-8") as file:
    faqs = json.load(file)

questions = [faq["question"] for faq in faqs]
answers = [faq["answer"] for faq in faqs]

# Train the text matching model
vectorizer = TfidfVectorizer()
question_vectors = vectorizer.fit_transform(questions)

# User input
user_question = st.text_input("Enter your question:")

if st.button("Get Answer"):
    if user_question.strip():
        user_vector = vectorizer.transform([user_question])
        similarities = cosine_similarity(user_vector, question_vectors)

        best_match = similarities.argmax()
        confidence = similarities[0][best_match]

        if confidence > 0.15:
            st.success("Chatbot Answer:")
            st.write(answers[best_match])
            st.caption("Matched FAQ: " + questions[best_match])
        else:
            st.warning("Sorry, I don't know the answer. Please try another question.")
    else:
        st.warning("Please enter a question.")

st.divider()
st.caption("Built with Python, Streamlit and NLP.")