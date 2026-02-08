"""
API 2: Static Demo Data (Fast, No Model Loading)
Port: 8001
Use this for quick testing and demos without waiting for model loading
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="SeismicGuard Demo API - Static Data")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# RESPONSE MODELS
# =========================
class SeismicPrediction(BaseModel):
    detection_type: str
    confidence: float
    model_accuracy: float
    description: str

class AnalysisResult(BaseModel):
    detection_type: str
    confidence: int
    tsunami_risk: str
    ai_description: str

# =========================
# ENDPOINTS
# =========================
@app.get("/")
async def root():
    return {
        "status": "ok", 
        "message": "Static Demo API - Port 8001",
        "note": "Returns static data for fast testing. Use api1.py (port 8000) for real models."
    }

@app.post("/predict-seismic-demo")
async def predict_seismic_demo(trace_index: int = 0):
    """
    CNN Demo - Returns static result (no model loading)
    """
    print(f"� Static CNN Demo - Trace {trace_index}")
    
    return SeismicPrediction(
        detection_type="Seismic Event",
        confidence=94.2,
        model_accuracy=97.3,
        description=f"Demo trace #{trace_index} (Magnitude 3.8). The 1D CNN analyzed the 3-channel waveform (Z, N, E components) at the P-wave arrival window (sample 4200). Characteristic seismic patterns detected indicating earthquake activity with high confidence."
    )

@app.post("/predict-seismic")
async def predict_seismic_upload():
    """
    CNN File Upload - Returns static result
    """
    print(f"📊 Static CNN Upload Demo")
    
    return SeismicPrediction(
        detection_type="Seismic Event",
        confidence=91.8,
        model_accuracy=97.3,
        description="Deep learning model detected a seismic event with high confidence. The 1D CNN analyzed the 3-channel waveform and identified characteristic patterns associated with earthquake activity."
    )

@app.post("/analyze-demo")
async def analyze_demo(trace_index: int = 0):
    """
    Gemini Demo - Returns static result
    """
    print(f"🤖 Static Gemini Demo - Trace {trace_index}")
    
    return AnalysisResult(
        detection_type="Earthquake",
        confidence=89,
        tsunami_risk="Medium",
        ai_description="Magnitude 3.8 seismic event detected. Spectrogram shows sustained low-frequency energy (1-20 Hz) with gradual onset characteristic of tectonic activity. Waveform analysis indicates typical earthquake patterns with clear P-wave arrival and sustained energy distribution. Moderate tsunami risk assessment based on magnitude and depth."
    )

@app.post("/analyze")
async def analyze_upload():
    """
    Gemini File Upload - Returns static result
    """
    print(f"🤖 Static Gemini Upload Demo")
    
    return AnalysisResult(
        detection_type="Earthquake",
        confidence=87,
        tsunami_risk="Medium",
        ai_description="Gemini AI analyzed the multimodal inputs (spectrogram + audio + physics features). The analysis shows characteristic earthquake patterns with sustained low-frequency energy and gradual onset. RMS confidence and duration support earthquake classification with moderate tsunami risk."
    )

if __name__ == "__main__":
    import uvicorn
    print("=" * 60)
    print("🚀 API 2: Static Demo Data")
    print("=" * 60)
    print("Port: 8001")
    print("Purpose: Fast testing with static results")
    print("Note: Use api1.py (port 8000) for real CNN + Gemini models")
    print("=" * 60)
    uvicorn.run(app, host="0.0.0.0", port=8001)
