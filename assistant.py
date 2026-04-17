from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import ollama

import pytesseract
from PIL import Image
import fitz
import re

# -----------------------------
# Tesseract path
# -----------------------------
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


# -----------------------------
# LOAD MODEL + FAISS DB
# -----------------------------
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = FAISS.load_local(
    "scheme_index",
    embedding_model,
    allow_dangerous_deserialization=True
)

print("🚀 YojanaMitra AI Ready!")


# -----------------------------
# LANGUAGE DETECTION
# -----------------------------
def get_language_instruction(query):

    if any('\u0C80' <= c <= '\u0CFF' for c in query):
        return "RESPOND ONLY IN KANNADA. DO NOT MIX LANGUAGES."
    elif any('\u0900' <= c <= '\u097F' for c in query):
        return "RESPOND ONLY IN HINDI. DO NOT MIX LANGUAGES."
    else:
        return "RESPOND ONLY IN ENGLISH. DO NOT MIX LANGUAGES."


# -----------------------------
# FILE EXTRACTION
# -----------------------------
def extract_pdf_text(file_path):
    text = ""
    doc = fitz.open(file_path)
    for page in doc:
        text += page.get_text()
    return text


def extract_image_text(file_path):
    img = Image.open(file_path)
    return pytesseract.image_to_string(img)


# -----------------------------
# CLEAN TEXT (IMPORTANT FIX)
# -----------------------------
def clean_text(text):
    text = text.replace("Ã¢â€šÂ¹", "₹")
    text = text.replace("Rs.", "₹")
    text = text.replace("Rs", "₹")

    # remove broken encoding artifacts
    text = re.sub(r'[^\x00-\x7F₹\n:.,()-]', ' ', text)

    return text


# -----------------------------
# MAIN FUNCTION
# -----------------------------
def get_scheme_answer(query, income, state, education, category, file_text=""):

    if file_text is None:
        file_text = ""

    language_instruction = get_language_instruction(query)

    # -----------------------------
    # IMPROVED SEARCH QUERY
    # -----------------------------
    search_query = f"{query} {state} {education} {category} {file_text}"

    # -----------------------------
    # FAISS SEARCH WITH SCORE FILTER
    # -----------------------------
    docs_with_score = db.similarity_search_with_score(search_query, k=5)

    filtered_docs = [doc for doc, score in docs_with_score if score < 0.72]

    # STATE FILTER (STRICT)
    filtered_docs = [
        doc for doc in filtered_docs
        if state.lower() in doc.page_content.lower()
    ]

    # -----------------------------
    # IF NOTHING FOUND
    # -----------------------------
    if len(filtered_docs) == 0:
        return "No suitable scheme found"

    context = "\n".join([d.page_content for d in filtered_docs])


    # -----------------------------
    # PROMPT (STRICT + CLEAN OUTPUT)
    # -----------------------------
    prompt = f"""
YOU ARE YOJANAMITRA AI (INDIAN GOVERNMENT SCHEME ASSISTANT)

🚨 STRICT RULES:
- Use ONLY dataset information
- NEVER invent schemes
- NEVER hallucinate
- NEVER include unrelated state schemes
- If no match exists, say EXACTLY: "No suitable scheme found"
- DO NOT add extra explanation outside format

{language_instruction}

User Profile:
Income: {income}
State: {state}
Education: {education}
Category: {category}

Dataset Context:
{context}

User Question:
{query}

OUTPUT FORMAT:

Scheme Name:
Eligibility:
Benefits:
Documents Required:
Application Steps:
Deadline:

FORMATTING RULES:
- Always use ₹ for money
- If Application Steps missing:
  "Please visit the official government website for application steps."

- If Deadline missing:
  "Please visit the official government website for deadline details."
"""


    # -----------------------------
    # LLM CALL (LOW TEMPERATURE)
    # -----------------------------
    response = ollama.chat(
        model="llama3",
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0.1}
    )

    answer = response["message"]["content"]

    # -----------------------------
    # CLEAN OUTPUT (IMPORTANT)
    # -----------------------------
    answer = clean_text(answer)

    # remove accidental bad phrase leakage
    answer = answer.replace("No suitable scheme found for", "").strip()

    # final safety check
    if len(filtered_docs) == 0:
        return "No suitable scheme found"

    return answer
