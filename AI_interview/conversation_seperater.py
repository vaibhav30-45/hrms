import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# -----------------------------
# IMPORTS
# -----------------------------
import re
import json


# -----------------------------
# JSON CLEANER
# -----------------------------
def extract_json(text: str):
    try:
        match = re.search(r"\[.*\]", text, re.DOTALL)

        if not match:
            print("❌ No JSON found")
            return []

        raw_json = match.group()
        raw_json = raw_json.replace("\n", " ")

        # Fix broken quotes inside words
        raw_json = re.sub(r'(\w)"(\w)', r"\1'\2", raw_json)

        # Fix unquoted keys
        raw_json = re.sub(
            r'([{,]\s*)(\w+)(\s*:)',
            r'\1"\2"\3',
            raw_json
        )

        return json.loads(raw_json)

    except Exception as e:
        print(f"❌ JSON PARSE ERROR: {e}")
        print("\n🔴 BROKEN JSON:\n")
        print(raw_json[:3000])
        return []


# -----------------------------
# MAIN FUNCTION (OPENAI)
# -----------------------------
def convert_to_json(transcript: str):

    prompt = f"""
You are a strict data structuring system.

Convert this interview into a Python-style JSON list.

RULES:
- Output ONLY valid JSON
- No markdown
- No explanation
- No extra text

FORMAT MUST BE EXACT:
[
  {{"speaker": "Interviewer", "text": "..."}},
  {{"speaker": "candidate", "text": "..."}},
  {{"speaker": "Interviewer", "text": "..."}}
]

CONVERSATION:
{transcript}
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",   # fast + cheap + good for structured output
        messages=[
            {"role": "system", "content": "You strictly output valid JSON only."},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )

    raw = response.choices[0].message.content

    parsed = extract_json(raw)

    if parsed:
        return parsed

    return []