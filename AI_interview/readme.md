# 🎯 AI Interview Evaluation System

An end-to-end AI-powered system that:
- Transcribes interview audio/video
- Separates interviewer & candidate conversations
- Processes and structures transcript
- Evaluates candidate performance using LLM (OpenAI)
- Generates final hiring recommendation

---

# 🚀 Features

✅ Video/Audio → Text Transcription (Whisper)  
✅ Automatic Speaker Separation (LLM-based)  
✅ Transcript Cleaning & Structuring  
✅ Multi-Dimensional Candidate Evaluation:
- Technical Skills
- Communication Skills
- Behavioral Analysis
- Job Description Match (optional)

✅ Final Hiring Recommendation:
- HIRE
- HOLD
- REJECT

---

# 📂 Project Structure


project/
│── main.py
│── conversation_seperater.py
│── evaluate_candidate.py
│── transcript_module.py
│── transcription_processing.py
│── input_data.py
│── .env


---

# ⚙️ Installation

## 1. Install Dependencies

```bash
pip install openai python-dotenv faster-whisper imageio-ffmpeg aiofiles
2. Setup Environment Variables

Create a .env file:

OPENAI_API_KEY=your_openai_api_key_here
🧠 System Workflow
🔹 Step 1: Transcription
Converts video/audio → text using Faster-Whisper
Extracts timestamps + segments

📁 File: transcript_module.py

🔹 Step 2: Conversation Separation
Uses OpenAI to classify:
Interviewer
Candidate

📁 File: conversation_seperater.py

🔹 Step 3: Transcript Processing
Cleans text
Extracts:
Q&A pairs
Candidate responses
Analysis inputs

📁 File: transcription_processing.py

🔹 Step 4: Candidate Evaluation (LLM)

Evaluates:

1. Technical Score
Concept clarity
Depth
Practical understanding
2. Communication Score
Grammar
Fluency
Vocabulary
3. Behavioral Score
Confidence
Professionalism
Attitude
4. JD Match Score (Optional)
Skill alignment
Relevance

📁 File: evaluate_candidate.py

🔹 Step 5: Final Score Calculation
With JD:
Final Score =
    30% Technical +
    25% Communication +
    20% Behavior +
    25% JD Match
Without JD:
Final Score =
    40% Technical +
    30% Communication +
    30% Behavior
🔹 Step 6: Recommendation Logic
Score Range	Decision
≥ 75	HIRE
55 – 74	HOLD
< 55	REJECT
▶️ Usage
Run Full Pipeline
python main.py
Output Example
{
  "technical_score": 72.5,
  "communication_score": 65.2,
  "behavior_score": 70.1,
  "jd_match_score": 68.0,
  "final_score": 69.8,
  "recommendation": "HOLD"
}
📥 Input Handling

Supports:

🎥 Video files (.mp4)
🎧 Audio files (.wav)

Uploads handled via:

Interview files
Resume files
Job descriptions

📁 File: input_data.py