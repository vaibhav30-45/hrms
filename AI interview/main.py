import sys
import time
from datetime import datetime
import requests

from transcript_module import process_interview_transcription 
from conversation_seperater import convert_to_json
from transcription_processing import process_transcript
from evaluate_candidate import evaluator


def log(step, msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] 🔹 {step}: {msg}", flush=True)


def run_pipeline():

    path = fr"C:\Users\Anurag Lawaniya\Downloads\Pre-Cas interview LSBU Part-1 __ CAS interview UK-2024🎗🏆- #CAS #interview #lsbu.mp4"

    log("STEP 1", "Extracting text from audio/video...")

    text = process_interview_transcription(
        media_path=path,
        is_video=True,
        device="cpu",
        model_size="small"
    )

    log("STEP 1", "Extraction completed ✔")

    # ----------------------------

    log("STEP 2", "Separating interviewer and candidate responses...")

    transcript = text.get("transcript", "")
    seperated_text = convert_to_json(transcript)

    log("STEP 2", "Separation completed ✔")

    # ----------------------------

    log("STEP 3", "Cleaning and processing transcript...")

    processed_text = process_transcript(seperated_text)

    log("STEP 3", "Processing completed ✔")

    # ----------------------------

    log("STEP 4", "Evaluating candidate performance...")

    result = evaluator.calculate_final_score(processed_text)

    log("STEP 4", "Evaluation completed ✔")

    # ---------------------------- FINAL OUTPUT

    log("FINAL", "================ FINAL RESULT ================")
    print(result, flush=True)
    log("FINAL", "Pipeline completed successfully ✔")


if __name__ == "__main__":
    run_pipeline()