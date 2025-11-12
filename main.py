import pandas as pd
import spacy
import streamlit as st
from sentence_transformers import SentenceTransformer, util

# Load models
nlp = spacy.load("en_core_web_sm")
model = SentenceTransformer('all-MiniLM-L6-v2')

# Load job dataset
jobs_df = pd.read_csv("data/jobs.csv")

# Text preprocessing function
def preprocess_text(text):
    doc = nlp(text.lower())
    return " ".join([token.lemma_ for token in doc if not token.is_stop and token.is_alpha])

# Job recommendation function
def recommend_jobs(resume_text, top_n=5):
    clean_resume = preprocess_text(resume_text)
    resume_embedding = model.encode(clean_resume, convert_to_tensor=True)

    jobs_df['similarity'] = jobs_df['Description'].apply(
        lambda desc: util.cos_sim(
            resume_embedding, model.encode(preprocess_text(desc), convert_to_tensor=True)
        ).item()
    )

    top_jobs = jobs_df.sort_values(by='similarity', ascending=False).head(top_n)
    return top_jobs[['Job Title', 'Company', 'Location', 'Skills Required', 'Experience', 'Salary', 'similarity']]

# Streamlit UI
st.title("💼 AI Job Recommender System")
st.write("Get the best job matches based on your resume or skills!")

resume_input = st.text_area("📝 Paste your resume or list your skills here:")

if st.button("Find Matching Jobs"):
    if resume_input.strip():
        with st.spinner("🔍 Finding best matches..."):
            recommendations = recommend_jobs(resume_input)
        st.success("✅ Here are your top job matches:")
        st.dataframe(recommendations)
    else:
        st.warning("⚠️ Please enter your resume or skills first.")



