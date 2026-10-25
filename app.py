import streamlit as st
import PyPDF2
from google import genai
from google.genai import types

# --- PAGE SETUP ---
st.set_page_config(page_title="AI Resume Tailor", page_icon="📄")
st.title("AI Resume Tailor (ATS Optimizer)")
st.write("Upload your master resume and paste a job description. The AI will rewrite your bullet points to match the required keywords without hallucinating.")

# --- HELPER FUNCTION ---
def extract_text_from_pdf(pdf_file):
    """Extracts text from an uploaded PDF file."""
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted
    return text

# --- UI INPUTS ---
uploaded_file = st.file_uploader("1. Upload Master Resume (PDF)", type=["pdf"])
job_description = st.text_area("2. Paste Job Description Here", height=200)

# --- PROCESSING LOGIC ---
if st.button("Optimize My Resume"):
    if not uploaded_file:
        st.error("Please upload a resume PDF.")
    elif not job_description:
        st.error("Please paste the job description.")
    else:
        with st.spinner("Analyzing keywords and rewriting bullet points..."):
            try:
                # 1. Extract text
                resume_text = extract_text_from_pdf(uploaded_file)
                
                # 2. Initialize Gemini Client securely using Streamlit Secrets
                client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])
                
                # 3. Define the strict rules (System Instruction)
                system_instruction = """
                You are an expert Technical Recruiter and ATS Optimization Specialist. Your job is to rewrite a candidate's resume bullet points to naturally include missing keywords from a target job description.

                CRITICAL RULES:
                1. NEVER invent metrics, years of experience, or skills the candidate did not state.
                2. DO NOT use fluffy buzzwords (e.g., "ninja", "spearheaded").
                3. Use the XYZ Formula: "Accomplished [X] as measured by [Y], by doing [Z]".
                4. Output ONLY the rewritten bullet points in Markdown format. No intro text.
                """
                
                # 4. Define the User Prompt
                prompt = f"""
                ORIGINAL RESUME:
                {resume_text}
                
                TARGET JOB DESCRIPTION:
                {job_description}
                
                TASK: Extract the core bullet points from the resume and rewrite them to highlight the skills and keywords demanded in the job description.
                """
                
                # 5. Call the API
                response = client.models.generate_content(
                    model='gemini-3.8-flash',
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=system_instruction,
                        temperature=0.2, # Low temperature keeps it factual and reduces hallucinations
                    ),
                )
                
                # 6. Display Output
                st.success("Resume Optimized!")
                st.markdown("### Your New Bullet Points")
                st.write(response.text)
                
            except Exception as e:
                st.error(f"An error occurred: {e}")