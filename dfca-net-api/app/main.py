from fastapi import FastAPI, File, UploadFile, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from typing import Literal
import traceback
import os

from .audio_processor import process_audio
from .predictor import predictor

# Response Models
class HealthResponse(BaseModel):
    message: str = Field(..., description="Welcome message and API status")

class PredictionResponse(BaseModel):
    filename: str = Field(..., description="Name of the uploaded audio file")
    prediction: Literal["Normal", "Abnormal"] = Field(..., description="Anomaly detection result")
    confidicence: str = Field(..., description="Confidence score (0.0000 to 1.0000)")

class ErrorResponse(BaseModel):
    detail: str = Field(..., description="Error message describing what went wrong")

# API Metadata
app = FastAPI(
    title="DFCA-Net Industrial Machine Anomaly Detection API",
    description="""
    ## 🔧 Industrial Machine Anomaly Detection using Deep Learning
    
    The DFCA-Net API provides state-of-the-art anomaly detection for industrial machinery using audio signal analysis. 
    Our Dual Frequency Cross-Attention Network (DFCA-Net) combines STFT and CQT spectrograms to detect abnormal 
    machine behavior with high accuracy.
    
    ### 🎯 Key Features:
    - **Real-time Detection**: Get instant anomaly predictions
    - **High Accuracy**: Advanced deep learning model with cross-attention mechanisms
    - **Audio Processing**: Supports .wav audio files with automatic preprocessing
    - **Confidence Scoring**: Provides confidence levels for each prediction
    - **Industrial Grade**: Designed for production industrial environments
    
    ### 🚀 How it Works:
    1. **Upload**: Submit a .wav audio file of your machine
    2. **Process**: Audio is converted to STFT and CQT spectrograms
    3. **Analyze**: DFCA-Net model processes the spectrograms
    4. **Predict**: Returns Normal/Abnormal classification with confidence
    
    ### 📋 Supported Audio:
    - **Format**: .wav files only
    - **Duration**: 1-10 seconds recommended
    - **Sample Rate**: Any (automatically resampled to 16kHz)
    - **File Size**: Up to 50MB (< 10MB recommended for speed)
    
    ### 🔬 Model Architecture:
    - **STFT Processing**: 512 FFT size, 64 mel bands
    - **CQT Processing**: 84 frequency bins, 36 bins per octave
    - **Fusion**: 256-dimensional cross-attention dual frequency fusion
    - **Classification**: Binary anomaly detection with 0.65 threshold
    """,
    version="1.0.0",
    contact={
        "name": "DFCA-Net Development Team",
        "email": "support@dfca-net.com",
    },
    license_info={
        "name": "MIT License",
        "url": "https://opensource.org/licenses/MIT",
    },
    servers=[
        {
            "url": "http://localhost:8000",
            "description": "Development server"
        }
    ],
    tags_metadata=[
        {
            "name": "Health Check",
            "description": "API status and health monitoring endpoints"
        },
        {
            "name": "Anomaly Detection",
            "description": "Core machine learning prediction endpoints for anomaly detection"
        }
    ],
    openapi_tags=[
        {
            "name": "Health Check",
            "description": "API status and health monitoring endpoints",
        },
        {
            "name": "Anomaly Detection", 
            "description": "Core machine learning prediction endpoints for anomaly detection",
        }
    ]
)

# Mount static files
static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # React dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get(
    "/",
    response_model=HealthResponse,
    tags=["Health Check"],
    summary="API Health Check",
    description="Returns the API status and welcome message. Use this endpoint to verify the API is running correctly.",
    responses={
        200: {
            "description": "API is running successfully",
            "content": {
                "application/json": {
                    "example": {
                        "message": "Welcome to the DFCA-Net Anomaly Detection API. Use the /docs endpoint to see the API documentation"
                    }
                }
            }
        }
    }
)
def read_root():
    """
    ## API Health Check
    
    Returns a welcome message confirming the API is operational.
    
    **Use this endpoint to:**
    - Verify API connectivity
    - Check if the service is running
    - Get basic API information
    """
    return {"message": "Welcome to the DFCA-Net Anomaly Detection API. Use the /docs endpoint to see the API documentation"}

@app.post(
    "/predict/",
    response_model=PredictionResponse,
    tags=["Anomaly Detection"],
    summary="Detect Machine Anomalies",
    description="Upload a .wav audio file to detect anomalies in industrial machine sounds using DFCA-Net deep learning model.",
    responses={
        200: {
            "description": "Successful anomaly detection prediction",
            "content": {
                "application/json": {
                    "examples": {
                        "normal_machine": {
                            "summary": "Normal Machine Sound",
                            "description": "Example response for a healthy machine",
                            "value": {
                                "filename": "normal_machine.wav",
                                "prediction": "Normal",
                                "confidicence": "0.8542"
                            }
                        },
                        "abnormal_machine": {
                            "summary": "Abnormal Machine Sound",
                            "description": "Example response for a faulty machine",
                            "value": {
                                "filename": "faulty_bearing.wav",
                                "prediction": "Abnormal",
                                "confidicence": "0.9123"
                            }
                        }
                    }
                }
            }
        },
        400: {
            "description": "Invalid file format or bad request",
            "model": ErrorResponse,
            "content": {
                "application/json": {
                    "examples": {
                        "invalid_format": {
                            "summary": "Invalid File Format",
                            "value": {
                                "detail": "Invalid file type. Please upload a .wav file."
                            }
                        }
                    }
                }
            }
        },
        422: {
            "description": "Validation error - missing file parameter",
            "content": {
                "application/json": {
                    "example": {
                        "detail": [
                            {
                                "loc": ["body", "file"],
                                "msg": "field required",
                                "type": "value_error.missing"
                            }
                        ]
                    }
                }
            }
        },
        500: {
            "description": "Internal server error during processing",
            "model": ErrorResponse,
            "content": {
                "application/json": {
                    "examples": {
                        "processing_error": {
                            "summary": "Processing Error",
                            "value": {
                                "detail": "An error occurred during processing: Model inference failed"
                            }
                        },
                        "corrupted_audio": {
                            "summary": "Corrupted Audio File",
                            "value": {
                                "detail": "An error occurred during processing: Unable to decode audio file"
                            }
                        }
                    }
                }
            }
        }
    }
)
async def predict_anomaly(
    file: UploadFile = File(
        ...,
        description="Audio file in .wav format containing machine sounds to analyze",
        media_type="audio/wav"
    )
):
    """
    ## 🎵 Machine Anomaly Detection
    
    Upload an industrial machine audio file to detect potential anomalies using our advanced DFCA-Net model.
    
    ### 📋 Requirements:
    - **File Format**: .wav audio files only
    - **Duration**: 1-10 seconds recommended for optimal results
    - **Sample Rate**: Any sample rate (automatically resampled to 16kHz)
    - **File Size**: Maximum 50MB (under 10MB recommended for faster processing)
    - **Content**: Clear machine audio without excessive background noise
    
    ### 🔄 Processing Pipeline:
    1. **Audio Preprocessing**: Resampling to 16kHz and pre-emphasis filtering
    2. **STFT Generation**: Short-Time Fourier Transform with mel-scale filtering
    3. **CQT Generation**: Constant-Q Transform for harmonic analysis
    4. **Dual Frequency Fusion**: Cross-attention mechanism combines both frequency representations
    5. **Classification**: Binary prediction with confidence scoring
    
    ### 📊 Response Details:
    - **filename**: Original uploaded file name
    - **prediction**: "Normal" or "Abnormal" classification
    - **confidicence**: Confidence score from 0.0000 to 1.0000
    
    ### 💡 Interpretation:
    - **Normal**: Machine operating within expected parameters
    - **Abnormal**: Potential fault detected, inspection recommended
    - **High Confidence (>0.8)**: Very reliable prediction
    - **Low Confidence (<0.6)**: Consider retesting with clearer audio
    
    ### ⚡ Performance:
    - **Small files (<1MB)**: ~1-3 seconds
    - **Medium files (1-5MB)**: ~3-8 seconds  
    - **Large files (5-10MB)**: ~8-15 seconds
    
    *Processing time depends on hardware capabilities (CPU/GPU) and system load.*
    """
    if not file.filename.lower().endswith('.wav'): # type: ignore
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid file type. Please upload a .wav file."
        )
    
    try:
        audio_bytes = await file.read()
        
        # Process audio through DFCA-Net pipeline
        stft_spec, cqt_spec = process_audio(audio_bytes)
        
        # Get model prediction
        label, confidence = predictor.predict(stft_spec, cqt_spec)
        
        return {
            "filename": file.filename,
            "prediction": label,
            "confidicence": f"{confidence:.4f}"
        }
    
    except Exception as error:
        traceback.print_exc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred during processing: {str(error)}"
        )

# Custom OpenAPI schema modification
def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    
    # Use the original openapi method from FastAPI
    from fastapi.openapi.utils import get_openapi
    
    openapi_schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )
    
    # Add custom info
    openapi_schema["info"]["x-logo"] = {
        "url": "https://fastapi.tiangolo.com/img/logo-margin/logo-teal.png"
    }
    
    # Add custom server info
    openapi_schema["info"]["x-api-id"] = "dfca-net-api"
    
    # Add examples to components
    if "components" not in openapi_schema:
        openapi_schema["components"] = {}
    
    openapi_schema["components"]["examples"] = {
        "NormalMachineAudio": {
            "summary": "Normal Machine Sound",
            "description": "Audio file from a healthy industrial machine",
            "value": "normal_machine_sound.wav"
        },
        "AbnormalMachineAudio": {
            "summary": "Faulty Machine Sound", 
            "description": "Audio file from a machine with bearing issues",
            "value": "faulty_bearing_sound.wav"
        }
    }
    
    # Add custom extensions
    openapi_schema["x-tagGroups"] = [
        {
            "name": "API Operations",
            "tags": ["Health Check", "Anomaly Detection"]
        }
    ]
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi

# Add custom documentation endpoint
@app.get("/api-info", include_in_schema=False)
async def api_info():
    """Custom API information endpoint"""
    return HTMLResponse(content="""
    <!DOCTYPE html>
    <html>
    <head>
        <title>DFCA-Net API Information</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 2rem; }
            .header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 2rem; border-radius: 8px; text-align: center; margin-bottom: 2rem; }
            .section { background: #f8f9fa; padding: 1.5rem; border-radius: 8px; margin: 1rem 0; }
            .code { background: #e9ecef; padding: 0.5rem; border-radius: 4px; font-family: monospace; }
            .endpoint { background: #d4edda; padding: 1rem; border-left: 4px solid #28a745; margin: 1rem 0; }
            .method { background: #007bff; color: white; padding: 0.25rem 0.5rem; border-radius: 4px; font-size: 0.8rem; }
        </style>
    </head>
    <body>
        <div class="header">
            <h1>🔧 DFCA-Net API</h1>
            <p>Industrial Machine Anomaly Detection</p>
        </div>
        
        <div class="section">
            <h2>📋 Quick Start</h2>
            <p>Get started with the DFCA-Net API in just a few steps:</p>
            <ol>
                <li>Prepare a .wav audio file of your machine</li>
                <li>Use the <code>/predict/</code> endpoint to upload and analyze</li>
                <li>Receive instant anomaly detection results</li>
            </ol>
        </div>
        
        <div class="section">
            <h2>🚀 Available Endpoints</h2>
            
            <div class="endpoint">
                <h3><span class="method">GET</span> /</h3>
                <p><strong>Health Check</strong> - Verify API status</p>
                <div class="code">curl -X GET "http://localhost:8000/"</div>
            </div>
            
            <div class="endpoint">
                <h3><span class="method">POST</span> /predict/</h3>
                <p><strong>Anomaly Detection</strong> - Upload audio for analysis</p>
                <div class="code">curl -X POST "http://localhost:8000/predict/" -F "file=@machine_sound.wav"</div>
            </div>
        </div>
        
        <div class="section">
            <h2>📊 Model Information</h2>
            <ul>
                <li><strong>Architecture:</strong> Dual Frequency Cross-Attention Network (DFCA-Net)</li>
                <li><strong>Input:</strong> STFT + CQT spectrograms</li>
                <li><strong>Output:</strong> Binary classification (Normal/Abnormal)</li>
                <li><strong>Confidence:</strong> Probability score with 0.65 threshold</li>
            </ul>
        </div>
        
        <div class="section">
            <h2>🔗 Documentation Links</h2>
            <ul>
                <li><a href="/docs" target="_blank">Interactive API Documentation (Swagger UI)</a></li>
                <li><a href="/redoc" target="_blank">Alternative Documentation (ReDoc)</a></li>
                <li><a href="/openapi.json" target="_blank">OpenAPI Schema (JSON)</a></li>
            </ul>
        </div>
    </body>
    </html>
    """)