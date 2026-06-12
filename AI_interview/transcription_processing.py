import re


# =========================================================
# 1. CLEAN CONVERSATION
# =========================================================

def clean_conversation(conversation):

    cleaned = []

    for item in conversation:

        text = re.sub(r"\s+", " ", item["text"]).strip()

        cleaned.append({
            "speaker": item["speaker"],
            "text": text
        })

    return cleaned


# =========================================================
# 2. DETECT QUESTION (IMPROVED)
# =========================================================

def is_question(text):

    text_lower = text.lower()

    question_keywords = [
        "tell me",
        "explain",
        "what",
        "why",
        "how",
        "when",
        "where",
        "difference",
        "describe",
        "can you",
        "could you",
        "do you",
        "are you",
        "have you"
    ]

    if text.strip().endswith("?"):
        return True

    return any(k in text_lower for k in question_keywords)


# =========================================================
# 3. SPLIT QA (BASED ON SPEAKER + FLOW)
# =========================================================

def split_question_answers(conversation):

    qa_pairs = []
    current_question = None

    for item in conversation:

        speaker = item["speaker"]
        text = item["text"]

        # interviewer question
        if speaker.lower() == "interviewer" and is_question(text):
            current_question = text

        # candidate answer
        elif speaker.lower() == "candidate" and current_question:
            qa_pairs.append({
                "question": current_question,
                "answer": text
            })
            current_question = None

    return qa_pairs


# =========================================================
# 4. ANALYSIS CLEANERS
# =========================================================

def clean_for_confidence_analysis(text):
    text = re.sub(r"[^\w\s.,?!\-]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def clean_for_communication_analysis(text):
    text = re.sub(r"[^\w\s.,?!]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def clean_for_technical_analysis(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# =========================================================
# 5. GENERATE ANALYSIS INPUTS
# =========================================================

def generate_analysis_inputs(conversation):

    confidence_text = []
    communication_text = []
    technical_text = []

    for item in conversation:

        text = item["text"]

        confidence_text.append(clean_for_confidence_analysis(text))
        communication_text.append(clean_for_communication_analysis(text))
        technical_text.append(clean_for_technical_analysis(text))

    return {
        "confidence_analysis_input": " ".join(confidence_text),
        "communication_analysis_input": " ".join(communication_text),
        "technical_analysis_input": " ".join(technical_text)
    }


# =========================================================
# 6. EXTRACT CANDIDATE RESPONSES
# =========================================================

def extract_candidate_responses(conversation):

    return [
        item["text"]
        for item in conversation
        if item["speaker"].lower() == "candidate"
    ]


# =========================================================
# 7. STATISTICS
# =========================================================

def generate_conversation_statistics(conversation):

    total_words = 0

    for item in conversation:
        total_words += len(item["text"].split())

    return {
        "total_messages": len(conversation),
        "total_word_count": total_words,
        "avg_words_per_message": round(
    total_words / len(conversation), 2
) if conversation else 0    }


# =========================================================
# 8. MAIN PIPELINE
# =========================================================

def process_transcript(conversation):

    cleaned = clean_conversation(conversation)

    qa_pairs = split_question_answers(cleaned)

    analysis_inputs = generate_analysis_inputs(cleaned)

    candidate_responses = extract_candidate_responses(cleaned)

    stats = generate_conversation_statistics(cleaned)

    return {
        "cleaned_conversation": cleaned,
        "qa_pairs": qa_pairs,
        "candidate_responses": candidate_responses,
        "confidence_analysis_input": analysis_inputs["confidence_analysis_input"],
        "communication_analysis_input": analysis_inputs["communication_analysis_input"],
        "technical_analysis_input": analysis_inputs["technical_analysis_input"],
        "conversation_statistics": stats
    }


# =========================================================
# TEST
# =========================================================

# if __name__ == "__main__":

#     sample = [
#         {"speaker": "Interviewer", "text": "Yeah, so yes, so first of all, I just wanted to just to check your full name and your date of birth for me please. Well, my name is Tanbeh Khosain and my date of birth is 5th November 2003."},
#     {"speaker": "Interviewer", "text": "Okay, thank you. And what course are you studying?"},
#     {"speaker": "Candidate", "text": "I have applied for BSc honours in computer science."},
#     {"speaker": "Interviewer", "text": "Okay, that's fine, thank you. Okay, so let me just going to go do a little run through. So if would you be able to, are you alone in the room?"},
#     {"speaker": "Candidate", "text": "Yes, I am alone in my room."},
#     {"speaker": "Interviewer", "text": "Okay, would you be able to just show me around the room?"},
#     {"speaker": "Candidate", "text": "Okay. Yeah, if you just put your camera around. Ah, here it is. Is this okay?"},
#     {"speaker": "Interviewer", "text": "Yes, yes, just, yeah, if you just go around, yeah."},
#     {"speaker": "Candidate", "text": "Yeah, that's fine. Thank you very much. Thank you."},
#     {"speaker": "Interviewer", "text": "Okay, so I appreciate your time today to speaking with me. So the interview will be conducted as part of the university's precast process."},
#     {"speaker": "Interviewer", "text": "So during the precast process, we perform various checks to ensure that applicants meet all the visa requirements before getting the CAS. So these checks include reviewing financial evidence, supporting documents and assessing your general needs to studying on the course."},
#     {"speaker": "Interviewer", "text": "And this interview is part of the checks,"},
#     {"speaker": "Candidate", "text": "Okay."},
#     {"speaker": "Interviewer", "text": "So the interview will be about to approximately 15 to 20 minutes. So you shouldn't not be using any notes during the interview."},
#     {"speaker": "Interviewer", "text": "We will ask that you provide detailed answers to the questions."},
#     {"speaker": "Interviewer", "text": "And I'll be taking notes as you speak. So please speak clearly and just bear with me if I keep repeating myself. It's just that because I'm writing some notes down."},
#     {"speaker": "Candidate", "text": "Yes, okay. Yeah."},
#     {"speaker": "Interviewer", "text": "So if you have any questions before the interview or at the end, please feel free to ask, okay?"},
#     {"speaker": "Candidate", "text": "Yes, okay."},
#     {"speaker": "Interviewer", "text": "So just start off. How are you today? Are you okay?"},
#     {"speaker": "Candidate", "text": "Yes, I'm fine. What about you?"},
#     {"speaker": "Interviewer", "text": "Yes, I'm good. Thanks. Thank you for asking."},
#     {"speaker": "Interviewer", "text": "So the first question is at LSBU, we have many societies you could join. How do you usually like to spend your free time?"},
#     {"speaker": "Candidate", "text": "Well, I love to pass my leisure time with gossiping with my friends and playing cricket. I love to play cricket in my school life. I played cricket very much. And I was the captain in my cricket internal Premier League in school."},
#     {"speaker": "Candidate", "text": "And if I get any chance to play in LSBU sports club or sports centre, then I will obviously join this team and I will pass my leisure time by those kind of activities."},
#     {"speaker": "Interviewer", "text": "Okay, thank you."},
#     {"speaker": "Interviewer", "text": "Have you studied in the UK before?"},
#     {"speaker": "Candidate", "text": "No, this is my first time I am applying for."},
#     {"speaker": "Interviewer", "text": "Okay, that's fine."},
#     {"speaker": "Interviewer", "text": "Could you give me just a summary of your previous studies in your home country?"},
#     {"speaker": "Candidate", "text": "Well, previously I have completed my higher secondary certificate in 2021 from Meepu College and I got 4.50 from Science Background."},
#     {"speaker": "Candidate", "text": "There I have studied physics, chemistry, information, communication and technology."},
#     {"speaker": "Interviewer", "text": "Okay, hello. Yes, that's fine."},
#     {"speaker": "Interviewer", "text": "So my next question is, when did you finish your last study?"},
#     {"speaker": "Candidate", "text": "Well, as I said before, I have completed my high secondary school certificate in May, high secondary school certificate in December 2021."},
#     {"speaker": "Interviewer", "text": "Okay. Okay, that's fine."},
#     {"speaker": "Interviewer", "text": "My next question is, what course have you applied for?"},
#     {"speaker": "Candidate", "text": "I have applied for BSE honours in computer science."},
#     {"speaker": "Interviewer", "text": "Okay. Could you give me just some specific details about the course at LSBU?"},
#     {"speaker": "Candidate", "text": "Well, in London South Bank University, I have to complete some modules in first year,"},
#     {"speaker": "Candidate", "text": "which is fundamentals of computer science,"},
#     {"speaker": "Candidate", "text": "fundamentals of software development,"},
#     {"speaker": "Candidate", "text": "discrete mathematics,"},
#     {"speaker": "Candidate", "text": "professional practice,"},
#     {"speaker": "Candidate", "text": "data structure and algorithm,"},
#     {"speaker": "Candidate", "text": "requirements analysis and UCD."},
#     {"speaker": "Interviewer", "text": "Okay."},
#     {"speaker": "Candidate", "text": "And all of those are very interesting modules,"},
#     {"speaker": "Candidate", "text": "but among of them,"},
#     {"speaker": "Candidate", "text": "fundamentals of software developing seems much interesting."},
#     {"speaker": "Candidate", "text": "And it will be my favorite one."},
#     {"speaker": "Candidate", "text": "And by this module, I can learn the fundamentals of computer programming,"},
#     {"speaker": "Candidate", "text": "covering variables,"},
#     {"speaker": "Candidate", "text": "areas,"},
#     {"speaker": "Candidate", "text": "data types,"},
#     {"speaker": "Candidate", "text": "algorithms,"},
#     {"speaker": "Candidate", "text": "conditional and interactive code,"},
#     {"speaker": "Candidate", "text": "use of the functions."},
#     {"speaker": "Candidate", "text": "And I can also learn how to write simple programs,"},
#     {"speaker": "Candidate", "text": "making use of a contemporary programming language and the professional and developer environment development."},
#     {"speaker": "Candidate", "text": "And this module will be very helpful for me in my future career as a software engineer."},
#     {"speaker": "Interviewer", "text": "Okay, that sounds good. What about practical sessions or projects related to this course?"},
#     {"speaker": "Candidate", "text": "Yes, there will be practical sessions and projects related to this course."},
#     {"speaker": "Candidate", "text": "I have learned that we will work on mini-projects in teams during the course. These projects will help us to apply what we have learned in class to real-world scenarios."},
#     {"speaker": "Interviewer", "text": "Okay, that's good to hear. What about the assessment method for this course?"},
#     {"speaker": "Candidate", "text": "The assessment method for this course includes quizzes, assignments, and exams."},
#     {"speaker": "Candidate", "text": "There will also be presentations and group discussions to assess our communication skills and teamwork abilities."},
#     {"speaker": "Interviewer", "text": "Okay, that sounds comprehensive. So what are your expectations from studying at LSBU?"},
#     {"speaker": "Candidate", "text": "My main expectation from studying at LSBU is to gain a strong foundation in computer science and software engineering."},
#     {"speaker": "Candidate", "text": "I want to learn from experienced faculty members who are experts in their respective fields. I also expect to develop my problem-solving skills and work on challenging projects during the course."},
#     {"speaker": "Candidate", "text": "Furthermore, I hope to make connections with industry professionals and secure a good job after graduation."},
#     {"speaker": "Interviewer", "text": "That sounds like a solid plan. Thank you for your time today. We will be in touch with you soon regarding the next steps."},
#     {"speaker": "Candidate", "text": "Thank you very much for the opportunity. I am looking forward to hearing from you soon."}
#     ]

#     from pprint import pprint
#     pprint(process_transcript(sample))