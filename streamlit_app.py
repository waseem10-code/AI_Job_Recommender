
import streamlit as st
import pandas as pd
import spacy
from sentence_transformers import SentenceTransformer, util

# Load NLP and model
nlp = spacy.load("en_core_web_sm")
model = SentenceTransformer('all-MiniLM-L6-v2')

# Load job dataset
jobs_df = pd.read_csv("data/jobs.csv")

# Preprocess function
def preprocess_text(text):
    doc = nlp(text.lower())
    return " ".join([token.lemma_ for token in doc if not token.is_stop and token.is_alpha])

# Recommend jobs function
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

# ---------------- Streamlit UI ----------------
st.title("💼 AI Job Recommender System")

st.write("Upload your resume or paste your skills below to get top job matches!")

resume_input = st.text_area("Paste your resume or skills here:")

if st.button("Find Jobs"):
    if resume_input.strip():
        recommendations = recommend_jobs(resume_input)
        st.subheader("Top Job Recommendations:")
        st.dataframe(recommendations)
    else:
        st.warning("Please enter your resume text first!")





 

