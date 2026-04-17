# assistant.py

import os
import pandas as pd
from openai import OpenAI
import pytesseract
from PIL import Image
import fitz


# --------------------------------------------------
# TESSERACT PATH (change only if needed locally)
# --------------------------------------------------
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


# --------------------------------------------------
# GROK CLIENT SETUP
# --------------------------------------------------
client = OpenAI(
    api_key=os.getenv("XAI_API_KEY"),
    base_url="https://api.x.ai/v1"
)


print("🚀 YojanaMitra AI Ready (Using Grok Model)")


# --------------------------------------------------
# LANGUAGE DETECTION FUNCTION
# --------------------------------------------------
def get_language_instruction(query):

    if any('\u0C80' <= c <= '\u0CFF' for c in query):
        return "RESPOND ONLY IN KANNADA. DO NOT MIX LANGUAGES."

    elif any('\u0900' <= c <= '\u097F' for c in query):
        return "RESPOND ONLY IN HINDI. DO NOT MIX LANGUAGES."

    else:
        return "RESPOND ONLY IN ENGLISH. DO NOT MIX LANGUAGES."


# --------------------------------------------------
# FILE TEXT EXTRACTION (PDF)
# --------------------------------------------------
def extract_pdf_text(file_path):

    text = ""

    try:
        doc = fitz.open(file_path)

        for page in doc:
            text += page.get_text()

    except:
        text = ""

    return text


# --------------------------------------------------
# FILE TEXT EXTRACTION (IMAGE OCR)
# --------------------------------------------------
def extract_image_text(file_path):

    try:
        img = Image.open(file_path)
        return pytesseract.image_to_string(img)

    except:
        return ""


# --------------------------------------------------
# MAIN SCHEME RESPONSE FUNCTION
# --------------------------------------------------
def get_scheme_answer(
        query,
        income,
        state,
        education,
        category,
        file_text=""
):

    if file_text is None:
        file_text = ""

    language_instruction = get_language_instruction(query)


    # --------------------------------------------------
    # USER PROFILE CONTEXT
    # --------------------------------------------------
    user_context = f"""

User Profile:

Income: {income}
State: {state}
Education: {education}
Category: {category}

Uploaded Document Content:

{file_text}

"""


    # --------------------------------------------------
    # FINAL PROMPT FOR GROK
    # --------------------------------------------------
    prompt = f"""

YOU ARE YOJANAMITRA AI — AN INDIAN GOVERNMENT SCHEME ASSISTANT.

STRICT RULES:

1. Suggest ONLY REAL GOVERNMENT SCHEMES
2. Prefer schemes matching user state
3. Prefer student-specific schemes if education provided
4. Prefer disability schemes if category = disabled
5. NEVER hallucinate fake schemes
6. If unsure — say:

"No suitable scheme found"

{language_instruction}

{user_context}

User Question:

{query}


OUTPUT FORMAT STRICTLY:

🎯 Result

Scheme Name:
Eligibility:
Benefits:
Documents Required:
Application Steps:
Deadline:

"""


    # --------------------------------------------------
    # GROK MODEL CALL
    # --------------------------------------------------
    try:

        response = client.chat.completions.create(

            model="grok-2-latest",

            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],

            temperature=0.2

        )

        answer = response.choices[0].message.content.strip()


    except Exception as e:

        answer = "AI service temporarily unavailable. Please try again later."


    return answer
