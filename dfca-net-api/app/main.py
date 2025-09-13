from fastapi import FastAPI, File, UploadFile, HTTPException
import traceback

from .audio_processor import process_audio
from .predictor import predictor

app = FastAPI(title="Industrial Machine Anomaly Detection API")

@app.get("/")
def read_root():
    return {"message": "Welcome to the DFCA-Net Anomaly Detection API. Use the /docs endpoint to see the API documentation"}

@app.post("/predict/")
async def predict_anomaly(file: UploadFile = File(...)):
    """
        Accepts a .wav audio file, process it, and returns the anomaly prediction
    """
    if not file.filename.lower().endswith('.wav'): # type: ignore
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a .wav file.")
    try:
        audio_bytes = await file.read()

        stft_spec, cqt_spec = process_audio(audio_bytes)

        label, confidence = predictor.predict(stft_spec, cqt_spec)

        return {
            "filename" : file.filename,
            "prediction": label,
            "confidicence": f"{confidence:.4f}"
        }
    
    except Exception as error:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"An error occurred during processing: {str(error)}")
