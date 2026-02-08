#!/usr/bin/env python3
"""Test the analyze endpoint to debug 500 errors"""
import os
import sys
from pathlib import Path

# Add current dir to path
sys.path.insert(0, str(Path(__file__).parent))

def test_basic_imports():
    """Test if all imports work"""
    print("Testing imports...")
    try:
        import librosa
        print("✓ librosa")
        import google.generativeai as genai
        print("✓ google.generativeai")
        import tempfile
        print("✓ tempfile")
        import numpy as np
        print("✓ numpy")
        from scipy.signal import butter, filtfilt
        print("✓ scipy")
        print("\n✅ All imports successful!\n")
        return True
    except Exception as e:
        print(f"\n❌ Import failed: {e}\n")
        return False

def test_file_exists():
    """Test if japan.wav exists"""
    print("Testing file access...")
    if Path("japan.wav").exists():
        size = Path("japan.wav").stat().st_size
        print(f"✓ japan.wav exists ({size:,} bytes)\n")
        return True
    else:
        print("❌ japan.wav not found\n")
        return False

def test_gemini_key():
    """Test Gemini API key"""
    print("Testing Gemini API key...")
    from dotenv import load_dotenv
    load_dotenv()
    key = os.getenv("GEMINI_API_KEY")
    if key:
        print(f"✓ GEMINI_API_KEY found (starts with: {key[:20]}...)\n")
        return True
    else:
        print("❌ GEMINI_API_KEY not found\n")
        return False

def test_seismic_candidate():
    """Test the seismic_candidate function"""
    print("Testing seismic_candidate function...")
    try:
        import librosa
        from scipy.signal import butter, filtfilt
        import numpy as np
        
        def bandpass(data, low, high, sr, order=4):
            nyq = 0.5 * sr
            b, a = butter(order, [low/nyq, high/nyq], btype="band")
            return filtfilt(b, a, data)
        
        # Load audio
        y, sr = librosa.load("japan.wav", sr=200, mono=True)
        duration = librosa.get_duration(y=y, sr=sr)
        
        # Filter
        y = bandpass(y, 1, 20, sr)
        
        # RMS
        win = int(sr * 2)
        rms = np.array([
            np.sqrt(np.mean(y[i:i+win]**2))
            for i in range(0, len(y)-win, win)
        ])
        
        rms = rms / np.max(rms)
        detected = (rms.max() > 0.6) and (np.sum(rms > 0.4) > 4)
        
        print(f"✓ RMS: {rms.max():.3f}, Duration: {duration:.2f}s, Detected: {detected}\n")
        return True, rms.max(), duration
    except Exception as e:
        print(f"❌ Error: {e}\n")
        import traceback
        traceback.print_exc()
        return False, 0, 0

def test_spectrogram():
    """Test spectrogram generation"""
    print("Testing spectrogram generation...")
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import librosa
        import librosa.display
        import io
        import numpy as np
        
        y, sr = librosa.load("japan.wav", sr=200)
        S = librosa.stft(y, n_fft=256)
        S_db = librosa.amplitude_to_db(np.abs(S), ref=np.max)
        
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(S_db, sr=sr, x_axis="time", y_axis="hz")
        plt.ylim(0, 50)
        plt.colorbar()
        plt.title("Filtered Spectrogram (0–50 Hz)")
        plt.tight_layout()
        
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=100)
        plt.close()
        buf.seek(0)
        spec_bytes = buf.read()
        
        print(f"✓ Spectrogram generated ({len(spec_bytes):,} bytes)\n")
        return True, spec_bytes
    except Exception as e:
        print(f"❌ Error: {e}\n")
        import traceback
        traceback.print_exc()
        return False, None

if __name__ == "__main__":
    print("=" * 50)
    print("ANALYZE ENDPOINT DEBUG TEST")
    print("=" * 50 + "\n")
    
    all_pass = True
    
    all_pass &= test_basic_imports()
    all_pass &= test_file_exists()
    all_pass &= test_gemini_key()
    
    success, rms, duration = test_seismic_candidate()
    all_pass &= success
    
    success, spec = test_spectrogram()
    all_pass &= success
    
    print("=" * 50)
    if all_pass:
        print("✅ ALL TESTS PASSED!")
        print("\nNow testing Gemini API call...")
        print("-" * 50)
        
        try:
            import google.generativeai as genai
            from dotenv import load_dotenv
            load_dotenv()
            
            genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
            
            # Read audio
            with open("japan.wav", "rb") as f:
                audio_bytes = f.read()
            
            prompt = f"""
Analyze this underwater audio. Classify as: Earthquake, Explosion, Marine Life, or Ambient Noise.
Respond ONLY with valid JSON:
{{
  "final_type": "Earthquake" | "Explosion" | "Marine Life" | "Ambient Noise",
  "confidence": <0-100>,
  "reason": "brief explanation"
}}
"""
            
            model = genai.GenerativeModel("gemini-2.0-flash-exp")
            response = model.generate_content([
                {"mime_type": "image/png", "data": spec},
                {"mime_type": "audio/wav", "data": audio_bytes},
                prompt
            ])
            
            print(f"Gemini response:\n{response.text}\n")
            
            # Try to parse
            import json
            text = response.text.strip().replace("```json", "").replace("```", "").strip()
            result = json.loads(text)
            print(f"✅ Parsed JSON: {result}")
            
        except Exception as e:
            print(f"❌ Gemini API call failed: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("❌ SOME TESTS FAILED")
    print("=" * 50)
