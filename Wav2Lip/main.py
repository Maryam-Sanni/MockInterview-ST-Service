from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from celery.result import AsyncResult
from celery_app import celery
from tasks import generate_video_task
import os

app = FastAPI()

OUTPUT_DIR = "results"
app.mount("/results", StaticFiles(directory=OUTPUT_DIR), name="results")


class GenerateRequest(BaseModel):
    text: str
    language: str = "en"


# 🚀 1. Submit job (FAST response)
@app.post("/generate")
def generate(req: GenerateRequest):
    task = generate_video_task.delay(req.text, req.language)

    return {
        "job_id": task.id,
        "status": "processing"
    }


# 📊 2. Check job status
@app.get("/status/{task_id}")
def get_status(task_id: str):
    task = AsyncResult(task_id, app=celery)

    if task.state == "PENDING":
        return {"status": "pending"}

    if task.state == "STARTED":
        return {"status": "processing"}

    if task.state == "SUCCESS":
        return task.result

    if task.state == "FAILURE":
        return {"status": "failed", "error": str(task.info)}

    return {"status": task.state}