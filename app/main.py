from fastapi import FastAPI, UploadFile, File
import tempfile
import os
from gap_filler import fill_gaps
from delayed_mixer import synth_with_delay

app = FastAPI()

@app.post("/gapfill")
async def process_gapfill(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(await file.read())
        input_path = tmp.name

    output_path = input_path.replace(".wav", "_gapfill.wav")
    result_path = fill_gaps(input_path, output_path)
    return {"message": "Gap-filled", "output": os.path.basename(result_path)}

@app.post("/delayedmix")
async def process_delayed_mix(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(await file.read())
        input_path = tmp.name

    output_path = input_path.replace(".wav", "_delayedmix.wav")
    result_path = synth_with_delay(input_path, output_path)
    return {"message": "Delayed mix complete", "output": os.path.basename(result_path)}

