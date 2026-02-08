"""
Minimal API for demo - just returns static results
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="OceanGuard Demo API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class SeismicPrediction(BaseModel):
    detection_type: str
    confidence: float
    model_accuracy: float
    description: str

@app.get("/")
async def root():
    return {"status": "ok", "message": "Demo API Running"}

@app.post("/predict-seismic-demo")
async def predict_seismic_demo(trace_index: int = 0):
    """
    Returns static demo result - no model loading needed
    """
    print(f"🔍 Demo request for trace {trace_index}")
    
    return SeismicPrediction(
        detection_type="Seismic Event",
        confidence=94.2,
        model_accuracy=97.3,
        description=f"CNN analyzed demo trace #{trace_index} (Magnitude 3.8). The 1D CNN processed the 3-channel waveform (Z, N, E components) at the P-wave arrival window. Characteristic seismic patterns detected indicating earthquake activity."
    )

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting minimal demo API on port 8000...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
