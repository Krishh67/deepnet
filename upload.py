import os
import sys
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
import google.generativeai as genai
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# --- CONFIGURATION ---
SAMPLE_RATE = 200   # Analysis Rate
SPECTROGRAM_IMG = "file_spectrogram.png"
# Lower threshold for file analysis? Keep higher for clean files.
DETECTION_THRESHOLD = 0.001 

def process_audio_file(file_path):
    """Checks for Low Frequency Energy Event."""
    print(f"Analyzing {file_path}...")
    try:
        y, sr = librosa.load(file_path, sr=SAMPLE_RATE)
    except Exception as e:
        print(f"Error loading audio: {e}")
        return False

    # A. Generate Spectrogram
    S = librosa.stft(y, n_fft=256)
    S_db = librosa.amplitude_to_db(np.abs(S), ref=np.max)

    plt.figure(figsize=(10, 4))
    librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='hz')
    plt.colorbar(format='%+2.0f dB')
    plt.title(f'Spectrogram: {os.path.basename(file_path)}')
    plt.tight_layout()
    plt.savefig(SPECTROGRAM_IMG)
    plt.close()
    print(f"-> Spectrogram saved to {SPECTROGRAM_IMG}")

    # B. Physics Filter (1-20Hz)
    def bandpass_filter(data, lowcut, highcut, fs, order=4):
        nyq = 0.5 * fs
        if lowcut >= nyq: lowcut = nyq - 1
        if highcut >= nyq: highcut = nyq - 0.1
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        return filtfilt(b, a, data)

    try:
        y_filt = bandpass_filter(y, 1.0, 20.0, sr)
    except ValueError:
        return False

    rms_val = np.sqrt(np.mean(y_filt**2))
    
    print(f"📊 Signal Energy (1-20Hz): {rms_val:.6f}", end="")
    
    if rms_val > DETECTION_THRESHOLD:
        print(" -> ⚠️ TRIGGERED! Event Detected.")
        return True
    else:
        print(" -> 💤 Quiet (Below Threshold)")
        return False

def classify_event(api_key):
    """Sends Spectrogram to Gemini."""
    if not os.path.exists(SPECTROGRAM_IMG): return

    print("☁️ Sending Spectrogram to Gemini AI...", end="", flush=True)
    
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.5-flash')
        
        img_data = Path(SPECTROGRAM_IMG).read_bytes()
        
        prompt = """
        Analyze this spectrogram image.
        Classify the sound source.
        Only choose from: [Earthquake, Marine Life, Explosion, Ambient Noise].
        Provide a 1-sentence reasoning.
        """
        
        response = model.generate_content([
            {'mime_type': 'image/png', 'data': img_data},
            prompt
        ])
        
        print(" Done!")
        print("\n" + "="*40)
        print(f"🤖 AI CLASSIFICATION: {response.text.strip()}")
        print("="*40 + "\n")
        
    except Exception as e:
        print(f"\n❌ AI Error: {e}")

if __name__ == "__main__":
    print("🌊 OceanGuard File Analyzer 🌊")
    print("------------------------------")
    
    # 1. API Key Check
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        api_key = input("Enter Google Gemini API Key: ").strip()
        if not api_key:
            print("No key provided. Exiting.")
            sys.exit()

    while True:
        print("\nInput file path (or 'q' to quit):")
        file_path = input("> ").strip().strip('"').strip("'") # Handle quotes from drag-and-drop
        
        if file_path.lower() == 'q':
            break

        if os.path.exists(file_path):
            is_event = process_audio_file(file_path)
            
            # Ask to classify if detected, or allow force check
            if is_event:
                print("Event detected based on physics check. Auto-classifying...")
                classify_event(api_key)
            else:
                user_force = input("No event detected. Force classify anyway? (y/n): ").strip().lower()
                if user_force == 'y':
                    classify_event(api_key)
        else:
            print("❌ File not found. Please try again.")

    print("Exiting.")
