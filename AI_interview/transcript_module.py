# =========================================================
# SIMPLE INTERVIEW TRANSCRIPTION MODULE
# =========================================================

# FEATURES:
# 1. Video -> Audio
# 2. Audio Transcription
# 3. Clean Transcript
# 4. Timestamp Segments
# 5. No HuggingFace Token
# 6. No WhisperX
# 7. No Diarization
# 8. Easy Windows Support

# =========================================================
# INSTALL
# =========================================================

# pip install faster-whisper imageio-ffmpeg

# =========================================================

import subprocess
import imageio_ffmpeg

from faster_whisper import WhisperModel


# =========================================================
# 1. EXTRACT AUDIO FROM VIDEO
# =========================================================

def extract_audio_from_video(
    video_path,
    output_audio_path="temp_audio.wav"
):

    ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()

    command = [

        ffmpeg_path,

        "-i",
        video_path,

        "-ar",
        "16000",

        "-ac",
        "1",

        output_audio_path,

        "-y"
    ]

    subprocess.run(

        command,

        stdout=subprocess.PIPE,

        stderr=subprocess.PIPE
    )

    return output_audio_path


# =========================================================
# 2. CLEAN TEXT
# =========================================================

def clean_text(text):

    text = text.replace("\n", " ")

    text = " ".join(text.split())

    return text.strip()


# =========================================================
# 3. TRANSCRIBE AUDIO
# =========================================================

def transcribe_audio(
    audio_path,
    model_size="base",
    device="cpu"
):

    # -----------------------------------------------------
    # LOAD MODEL
    # -----------------------------------------------------

    model = WhisperModel(

        model_size,
        device=device,
        compute_type="int8"
    )

    # -----------------------------------------------------
    # TRANSCRIBE
    # -----------------------------------------------------

    segments, info = model.transcribe(

        audio_path,

        beam_size=5
    )

    transcript_segments = []

    full_transcript = ""

    # -----------------------------------------------------
    # PROCESS SEGMENTS
    # -----------------------------------------------------

    for segment in segments:

        cleaned_text = clean_text(
            segment.text
        )

        segment_data = {

            "start":
                round(
                    segment.start,
                    2
                ),

            "end":
                round(
                    segment.end,
                    2
                ),

            "text":
                cleaned_text
        }

        transcript_segments.append(
            segment_data
        )

        full_transcript += (
            cleaned_text + " "
        )

    return {

        "language":
            info.language,

        "transcript":
            full_transcript.strip(),

        "segments":
            transcript_segments
    }


# =========================================================
# 4. CREATE CONVERSATION FORMAT
# =========================================================

def create_conversation(
    transcript_segments
):

    conversation = []

    for segment in transcript_segments:

        conversation.append({

            "start":
                segment["start"],

            "end":
                segment["end"],

            "text":
                segment["text"]
        })

    return conversation


# =========================================================
# 5. COMPLETE PIPELINE
# =========================================================

def process_interview_transcription(

    media_path,

    is_video=False,

    device="cpu",

    model_size="small"
):

    try:

        # -------------------------------------------------
        # EXTRACT AUDIO
        # -------------------------------------------------

        if is_video:

            audio_path = (
                extract_audio_from_video(
                    media_path
                )
            )

        else:
            audio_path = media_path

        # -------------------------------------------------
        # TRANSCRIBE AUDIO
        # -------------------------------------------------

        transcription_result = (
            transcribe_audio(

                audio_path,

                model_size=model_size,

                device=device
            )
        )

        # -------------------------------------------------
        # CREATE CONVERSATION
        # -------------------------------------------------

        conversation = (
            create_conversation(

                transcription_result[
                    "segments"
                ]
            )
        )

        return {

            "success": True,

            "language":
                transcription_result[
                    "language"
                ],

            "transcript":
                transcription_result[
                    "transcript"
                ],

            "conversation":
                conversation
        }

    except Exception as e:

        return {

            "success": False,

            "error": str(e)
        }


# =========================================================
# EXAMPLE USAGE
# =========================================================

# media_path = (
#     fr"C:\Users\Anurag Lawaniya\Downloads\Pre-Cas interview LSBU Part-1 __ CAS interview UK-2024🎗🏆- #CAS #interview #lsbu.mp4"
# )

# result = process_interview_transcription(

#     media_path=media_path,

#     is_video=True,

#     device="cpu",

#     model_size="small"
# )

# print(result["transcript"])