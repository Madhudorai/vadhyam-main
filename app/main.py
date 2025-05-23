from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
import os
import uuid
import tempfile

from gap_filler import fill_gaps
from delayed_mixer import synth_with_delay

app = FastAPI()

# Mount the static folder for serving index.html
static_path = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=static_path, html=True), name="static")

@app.get("/")
def serve_index():
    index_path = os.path.join(static_path, "index.html")
    with open(index_path, "r") as f:
        return HTMLResponse(content=f.read(), status_code=200)

# Job tracking dict
jobs = {}

@app.post("/upload")
async def upload_audio(background_tasks: BackgroundTasks, file: UploadFile = File(...), mode: str = "gapfill"):
    job_id = str(uuid.uuid4())
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(await file.read())
        input_path = tmp.name

    output_path = input_path.replace(".wav", f"_{mode}.wav")
    jobs[job_id] = {"status": "processing", "output": output_path}

    if mode == "gapfill":
        background_tasks.add_task(run_gapfill, job_id, input_path, output_path)
    elif mode == "delayedmix":
        background_tasks.add_task(run_delayedmix, job_id, input_path, output_path)
    else:
        return JSONResponse(status_code=400, content={"error": "Invalid mode"})

    return {"job_id": job_id}

@app.get("/status/{job_id}")
def check_status(job_id: str):
    if job_id not in jobs:
        return {"status": "not_found"}
    return {"status": jobs[job_id]["status"]}

@app.get("/output/{job_id}")
def get_output(job_id: str):
    if job_id not in jobs or jobs[job_id]["status"] != "done":
        return JSONResponse(status_code=404, content={"error": "Job not ready"})
    return FileResponse(jobs[job_id]["output"], media_type="audio/wav")

def run_gapfill(job_id, input_path, output_path):
    try:
        fill_gaps(input_path, output_path)
    finally:
        os.remove(input_path)
    jobs[job_id]["status"] = "done"

def run_delayedmix(job_id, input_path, output_path):
    try:
        synth_with_delay(input_path, output_path)
    finally:
        os.remove(input_path)
    jobs[job_id]["status"] = "done"
