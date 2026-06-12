import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# -----------------------------
# IMPORTS
# -----------------------------
import json
import re


class FinalScoreEvaluator:

    def __init__(self):
        pass

    # -----------------------------
    # LLM CALL (OPENAI)
    # -----------------------------
    def llm_score(self, prompt: str) -> float:
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",   # fast + cost-effective
                messages=[
                    {"role": "system", "content": "Return ONLY a numeric score between 0 and 100. No explanation."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0
            )

            output = response.choices[0].message.content.strip()

            # Try JSON parsing first
            try:
                data = json.loads(output)
                return float(data.get("score", 50))
            except:
                pass

            # Fallback: extract number
            match = re.search(r"\d+(\.\d+)?", output)
            return float(match.group()) if match else 50.0

        except Exception as e:
            print(f"❌ OpenAI Error: {e}")
            return 50.0


    # -----------------------------
    # TECHNICAL EVALUATION
    # -----------------------------
    def evaluate_technical(self, technical_input: str):
        prompt = f"""
You are an expert technical interviewer evaluating a candidate.

Evaluate TWO dimensions:

1. TECHNICAL KNOWLEDGE (0–100)
2. CONFIDENCE (0–100)

Return ONLY a number between 0 and 100 (average of both).

Transcript:
{technical_input}
"""
        return self.llm_score(prompt)


    # -----------------------------
    # COMMUNICATION
    # -----------------------------
    def evaluate_communication(self, communication_input: str):
        prompt = f"""
Score communication from 0 to 100.

Evaluate ONLY:
- grammar
- clarity
- fluency
- vocabulary

Do NOT evaluate confidence or technical skills.

Transcript:
{communication_input}

Return ONLY a number.
"""
        return self.llm_score(prompt)


    # -----------------------------
    # BEHAVIOR
    # -----------------------------
    def evaluate_behavior(self, behavioral_input: str):
        prompt = f"""
Score behavior from 0 to 100.

Evaluate:
- confidence
- professionalism
- attitude
- engagement

Do NOT evaluate grammar or technical knowledge.

Transcript:
{behavioral_input}

Return ONLY a number.
"""
        return self.llm_score(prompt)


    # -----------------------------
    # JD MATCH
    # -----------------------------
    def evaluate_jd_match(self, jd_input: str):
        prompt = f"""
Evaluate job description match (0–100).

Check:
- skill alignment
- relevance
- missing skills

Transcript:
{jd_input}

Return ONLY a number.
"""
        return self.llm_score(prompt)


    # -----------------------------
    # FINAL SCORE
    # -----------------------------
    def calculate_final_score(self, data: dict):

        jd_input = data.get("jd_input")

        tech = self.evaluate_technical(data["technical_analysis_input"])
        comm = self.evaluate_communication(data["communication_analysis_input"])
        beh = self.evaluate_behavior(data["confidence_analysis_input"])

        if jd_input:
            jd = self.evaluate_jd_match(jd_input)

            final_score = (
                tech * 0.30 +
                comm * 0.25 +
                beh * 0.20 +
                jd * 0.25
            )
        else:
            jd = 0.0

            final_score = (
                tech * 0.40 +
                comm * 0.30 +
                beh * 0.30
            )

        recommendation = self.get_recommendation(final_score)

        return {
            "technical_score": round(tech, 2),
            "communication_score": round(comm, 2),
            "behavior_score": round(beh, 2),
            "jd_match_score": round(jd, 2),
            "final_score": round(final_score, 2),
            "recommendation": recommendation
        }


    # -----------------------------
    # RECOMMENDATION
    # -----------------------------
    def get_recommendation(self, score):
        if score >= 75:
            return "HIRE"
        elif score >= 55:
            return "HOLD"
        else:
            return "REJECT"


# -----------------------------
# USAGE
# -----------------------------
evaluator = FinalScoreEvaluator()