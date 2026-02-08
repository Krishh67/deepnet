import os
import base64
import tempfile
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.generativeai as genai

# Load environment variables
load_dotenv()

# Initialize FastAPI
app = FastAPI(title="OceanGuard Audio Analysis API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in environment variables")

genai.configure(api_key=GEMINI_API_KEY)


class AnalysisResult(BaseModel):
    """Response model for analysis results"""
    detection_type: str
    confidence: int
    tsunami_risk: str
    ai_description: str


@app.get("/")
async def root():
    """Health check endpoint"""
    return {"status": "ok", "message": "OceanGuard Audio Analysis API"}


@app.post("/analyze", response_model=AnalysisResult)
async def analyze_audio(file: UploadFile = File(...)):
    """
    Analyze uploaded audio file using Gemini AI
    Returns: detection type, confidence, tsunami risk, and AI description
    """
    
    # Validate file type
    allowed_extensions = ['.wav', '.mp3']
    file_ext = Path(file.filename).suffix.lower()
    
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed: {', '.join(allowed_extensions)}"
        )
    
    try:
        # Save uploaded file to temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name
        
        # Read file as base64 for Gemini
        with open(tmp_path, 'rb') as f:
            audio_bytes = f.read()
            audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
        
        # Clean up temporary file
        os.unlink(tmp_path)
        
        # Prepare Gemini prompt
        prompt = """You are an expert in underwater acoustic signal interpretation and ocean sound analysis.

Analyze the provided audio (or its acoustic representation) and identify the most likely type of underwater acoustic event.

Your goal is to classify WHAT kind of event is present, if any, based on acoustic patterns such as energy distribution, duration, temporal structure, and frequency behavior.

IMPORTANT:
- Do NOT assume the event is an earthquake.
- Choose the classification that best matches the observed acoustic characteristics.
- If no clear event is present, classify it as Ambient Noise.

Respond with ONLY a valid JSON object.  
No markdown, no code blocks, no explanations outside JSON.

Return this exact structure:
{
  "detection_type": "<one of: Earthquake, Marine Life, Explosion, Ambient Noise>",
  "confidence": <integer between 0-100>,
  "tsunami_risk": "<one of: Low, Medium, High>",
  "ai_description": "<2-3 sentences describing what acoustic patterns are present and why this classification was chosen>"
}

Classification Guidelines (use as reference, not assumptions):

- Earthquake:
  Sustained low-frequency energy, long duration, gradual onset and decay, seismic-like acoustic behavior.

- Marine Life:
  Repetitive or rhythmic patterns, clicks, whistles, calls, or frequency-modulated sounds associated with biological activity.

- Explosion:
  Very sudden onset, short duration, strong broadband energy spike, impulsive acoustic signature.

- Ambient Noise:
  Diffuse or random energy, background ocean sounds, waves, distant activity, or no clearly distinguishable event.

Tsunami Risk Assessment:
- High:
  Only if a strong, sustained earthquake-like acoustic signature is present.
- Medium:
  If seismic-like features are present but weak, short, or ambiguous.
- Low:
  If the sound is marine life, explosion, ambient noise, or non-seismic in nature.

Be conservative and scientifically responsible in assigning tsunami risk.

Respond with ONLY the JSON object."""

        # Call Gemini API
        try:
            model = genai.GenerativeModel('gemini-2.5-flash')
            
            # Determine MIME type
            mime_type = 'audio/wav' if file_ext == '.wav' else 'audio/mpeg'
            
            response = model.generate_content([
                {
                    'mime_type': mime_type,
                    'data': audio_bytes
                },
                prompt
            ])
            
            # Parse JSON response
            import json
            response_text = response.text.strip()
            
            # Remove markdown code blocks if present
            if response_text.startswith('```'):
                lines = response_text.split('\n')
                response_text = '\n'.join(lines[1:-1]) if len(lines) > 2 else response_text
                response_text = response_text.replace('```json', '').replace('```', '').strip()
            
            gemini_result = json.loads(response_text)
            
            # Validate and prepare response
            result = AnalysisResult(
                detection_type=gemini_result.get("detection_type", "Ambient Noise"),
                confidence=int(gemini_result.get("confidence", 50)),
                tsunami_risk=gemini_result.get("tsunami_risk", "Low"),
                ai_description=gemini_result.get("ai_description", "Analysis completed")
            )
            
            return result
            
        except Exception as e:
            print(f"Gemini API Error: {e}")
            # Return fallback response on error
            return AnalysisResult(
                detection_type="Ambient Noise",
                confidence=50,
                tsunami_risk="Low",
                ai_description=f"Unable to classify audio due to error: {str(e)}"
            )
        
    except Exception as e:
        if 'tmp_path' in locals() and os.path.exists(tmp_path):
            os.unlink(tmp_path)
        
        raise HTTPException(
            status_code=500,
            detail=f"Error processing audio file: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
