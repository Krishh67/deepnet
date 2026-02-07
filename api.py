import os
import io
import base64
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import librosa
import librosa.display
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
import google.generativeai as genai
from dotenv import load_dotenv

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Load environment variables
load_dotenv()

# Initialize FastAPI
app = FastAPI(title="OceanGuard Audio Analysis API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Next.js frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
SAMPLE_RATE = 200
DETECTION_THRESHOLD = 0.001
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in environment variables")

genai.configure(api_key=GEMINI_API_KEY)


class AnalysisResult(BaseModel):
    """Response model for analysis results"""
    detection_type: str
    confidence: int
    tsunami_risk: str
    rms_energy: float
    frequency: float
    duration: float
    description: str
    reasoning: str
    spectrogram_base64: str
    is_event_detected: bool


def bandpass_filter(data: np.ndarray, lowcut: float, highcut: float, fs: int, order: int = 4) -> np.ndarray:
    """Apply bandpass filter to audio data"""
    nyq = 0.5 * fs
    if lowcut >= nyq:
        lowcut = nyq - 1
    if highcut >= nyq:
        highcut = nyq - 0.1
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)


def generate_spectrogram(y: np.ndarray, sr: int) -> str:
    """Generate spectrogram and return as base64 string"""
    S = librosa.stft(y, n_fft=256)
    S_db = librosa.amplitude_to_db(np.abs(S), ref=np.max)
    
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='hz')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Audio Spectrogram')
    plt.tight_layout()
    
    # Save to bytes buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    plt.close()
    buf.seek(0)
    
    # Convert to base64
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    return img_base64


def analyze_with_gemini(spectrogram_base64: str, rms_energy: float, duration: float) -> dict:
    """Use Gemini AI to classify the audio event from spectrogram"""
    
    # Decode base64 to bytes
    img_bytes = base64.b64decode(spectrogram_base64)
    
    prompt = f"""
Analyze this underwater acoustic spectrogram image.

**Audio Metrics:**
- RMS Energy (1-20Hz): {rms_energy:.6f}
- Duration: {duration:.2f} seconds

**Task:**
You must respond with ONLY a valid JSON object. Do not include any markdown formatting, code blocks, or extra text.

Return a JSON object with these exact fields:
{{
  "detection_type": "<one of: Earthquake, Marine Life, Explosion, Ambient Noise>",
  "confidence": <integer 0-100>,
  "tsunami_risk": "<one of: Low, Medium, High>",
  "description": "<brief 1-2 sentence description of what you see in the spectrogram>",
  "reasoning": "<1-2 sentences explaining your classification based on frequency patterns, energy distribution, and temporal characteristics>"
}}

Guidelines:
- Earthquake: Low-frequency (0-20Hz), sustained energy, gradual onset
- Marine Life: Varied frequencies, intermittent patterns, biological rhythms
- Explosion: Sudden spike, broadband energy, short duration
- Ambient Noise: Diffuse energy, no clear pattern, background activity

Respond with ONLY the JSON object, nothing else.
"""

    try:
        model = genai.GenerativeModel('gemini-2.0-flash-exp')
        response = model.generate_content([
            {'mime_type': 'image/png', 'data': img_bytes},
            prompt
        ])
        
        # Parse response text as JSON
        import json
        response_text = response.text.strip()
        
        # Remove markdown code blocks if present
        if response_text.startswith('```'):
            # Extract content between ``` markers
            lines = response_text.split('\n')
            response_text = '\n'.join(lines[1:-1]) if len(lines) > 2 else response_text
            response_text = response_text.replace('```json', '').replace('```', '').strip()
        
        result = json.loads(response_text)
        return result
        
    except Exception as e:
        print(f"Gemini API Error: {e}")
        # Return default values on error
        return {
            "detection_type": "Ambient Noise",
            "confidence": 50,
            "tsunami_risk": "Low",
            "description": "Unable to classify due to API error",
            "reasoning": f"Error occurred: {str(e)}"
        }


@app.get("/")
async def root():
    """Health check endpoint"""
    return {"status": "ok", "message": "OceanGuard Audio Analysis API"}


@app.post("/analyze", response_model=AnalysisResult)
async def analyze_audio(file: UploadFile = File(...)):
    """
    Analyze uploaded audio file for underwater seismic events
    
    Accepts: WAV, MP3, FLAC files
    Returns: Analysis results with detection type, confidence, and visualizations
    """
    
    # Validate file type
    allowed_extensions = ['.wav', '.mp3', '.flac', '.ogg']
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
        
        # Load audio file
        y, sr = librosa.load(tmp_path, sr=SAMPLE_RATE)
        duration = librosa.get_duration(y=y, sr=sr)
        
        # Generate spectrogram
        spectrogram_base64 = generate_spectrogram(y, sr)
        
        # Apply bandpass filter (1-20Hz for seismic events)
        try:
            y_filtered = bandpass_filter(y, 1.0, 20.0, sr)
        except ValueError as e:
            # If filtering fails, use original signal
            y_filtered = y
        
        # Calculate RMS energy
        rms_energy = float(np.sqrt(np.mean(y_filtered**2)))
        
        # Detect if event is present based on energy threshold
        is_event_detected = rms_energy > DETECTION_THRESHOLD
        
        # Get dominant frequency
        fft = np.fft.fft(y_filtered)
        freqs = np.fft.fftfreq(len(fft), 1/sr)
        magnitude = np.abs(fft)
        # Only consider positive frequencies
        positive_freqs = freqs[:len(freqs)//2]
        positive_magnitude = magnitude[:len(magnitude)//2]
        dominant_freq = float(positive_freqs[np.argmax(positive_magnitude)])
        
        # Use Gemini AI to classify the event
        gemini_result = analyze_with_gemini(spectrogram_base64, rms_energy, duration)
        
        # Clean up temporary file
        os.unlink(tmp_path)
        
        # Prepare response
        result = AnalysisResult(
            detection_type=gemini_result.get("detection_type", "Ambient Noise"),
            confidence=gemini_result.get("confidence", 50),
            tsunami_risk=gemini_result.get("tsunami_risk", "Low"),
            rms_energy=rms_energy,
            frequency=abs(dominant_freq),
            duration=duration,
            description=gemini_result.get("description", "Analysis completed"),
            reasoning=gemini_result.get("reasoning", "Classification based on acoustic patterns"),
            spectrogram_base64=spectrogram_base64,
            is_event_detected=is_event_detected
        )
        
        return result
        
    except Exception as e:
        # Clean up if file exists
        if 'tmp_path' in locals() and os.path.exists(tmp_path):
            os.unlink(tmp_path)
        
        raise HTTPException(
            status_code=500,
            detail=f"Error processing audio file: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
