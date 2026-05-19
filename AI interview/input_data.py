import os
import uuid
import aiofiles


async def upload_interview_file(file, upload_dir="uploads/interviews"):

    os.makedirs(upload_dir, exist_ok=True)

    extension = os.path.splitext(file.filename)[1]

    unique_name = f"{uuid.uuid4()}{extension}"

    file_path = os.path.join(upload_dir, unique_name)

    async with aiofiles.open(file_path, "wb") as f:
        content = await file.read()
        await f.write(content)

    return {
        "file_id": str(uuid.uuid4()),
        "file_name": unique_name,
        "file_path": file_path,
        "file_type": extension
    }


# =========================================================
# 2. Upload Resume
# =========================================================

async def upload_resume_file(file, upload_dir="uploads/resumes"):

    os.makedirs(upload_dir, exist_ok=True)

    extension = os.path.splitext(file.filename)[1]

    unique_name = f"{uuid.uuid4()}{extension}"

    file_path = os.path.join(upload_dir, unique_name)

    async with aiofiles.open(file_path, "wb") as f:
        content = await file.read()
        await f.write(content)

    return {
        "resume_id": str(uuid.uuid4()),
        "resume_name": unique_name,
        "resume_path": file_path
    }


# =========================================================
# 3. Upload Job Description
# =========================================================

async def upload_job_description(file, upload_dir="uploads/jd"):

    os.makedirs(upload_dir, exist_ok=True)

    extension = os.path.splitext(file.filename)[1]

    unique_name = f"{uuid.uuid4()}{extension}"

    file_path = os.path.join(upload_dir, unique_name)

    async with aiofiles.open(file_path, "wb") as f:
        content = await file.read()
        await f.write(content)

    return {
        "jd_id": str(uuid.uuid4()),
        "jd_name": unique_name,
        "jd_path": file_path
    }


# =========================================================
# 4. Save Interviewer Notes
# =========================================================

def save_interviewer_notes(
    interview_id,
    interviewer_name,
    notes,
    rating=None
):

    note_data = {
        "interview_id": interview_id,
        "interviewer_name": interviewer_name,
        "notes": notes,
        "rating": rating
    }

    return note_data