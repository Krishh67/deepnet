import numpy as np
import librosa
import tensorflow as tf
import tensorflow_hub as hub
from scipy.signal import butter, filtfilt

# =========================
# Load YAMNet (once)
# =========================
yamnet = hub.load("https://tfhub.dev/google/yamnet/1")

class_map = yamnet.class_map_path().numpy().decode("utf-8")
CLASS_NAMES = [l.strip() for l in open(class_map)]

MARINE_KEYWORDS = [
    "whale", "dolphin", "seal", "animal",
    "orca", "porpoise", "fish"
]

# =========================
# Signal processing (Earthquake candidate)
# =========================
def bandpass(data, low, high, sr, order=4):
    nyq = 0.5 * sr
    b, a = butter(order, [low/nyq, high/nyq], btype="band")
    return filtfilt(b, a, data)

def seismic_candidate(audio_path):
    # Load low-frequency version
    y, sr = librosa.load(audio_path, sr=200, mono=True)

    # Band-pass 1–20 Hz
    y = bandpass(y, 1, 20, sr)

    # RMS over 2-second windows
    win = int(sr * 2)
    rms = np.array([
        np.sqrt(np.mean(y[i:i+win]**2))
        for i in range(0, len(y)-win, win)
    ])

    rms = rms / np.max(rms)

    detected = (rms.max() > 0.6) and (np.sum(rms > 0.4) > 4)

    return detected, float(rms.max())

# =========================
# YAMNet: marine or not
# =========================
def yamnet_is_marine(audio_path, threshold=0.25):
    y, sr = librosa.load(audio_path, sr=16000, mono=True)
    scores, _, _ = yamnet(tf.convert_to_tensor(y, tf.float32))
    mean_scores = tf.reduce_mean(scores, axis=0).numpy()

    marine_score = 0.0
    for i, label in enumerate(CLASS_NAMES):
        if any(k in label.lower() for k in MARINE_KEYWORDS):
            marine_score += mean_scores[i]

    return marine_score > threshold, marine_score

# =========================
# FINAL DECISION (LOCK THIS)
# =========================
def detect_event(audio_path):
    seismic, rms_strength = seismic_candidate(audio_path)

    # If no seismic signature → stop early
    if not seismic:
        return {
            "detection": "Ambient Noise",
            "confidence": round(rms_strength, 2),
            "reason": "No sustained low-frequency seismic energy"
        }

    # Seismic candidate exists → check marine
    is_marine, marine_score = yamnet_is_marine(audio_path)

    if is_marine:
        return {
            "detection": "Marine Life",
            "confidence": round(marine_score, 2),
            "reason": "YAMNet indicates biological acoustic patterns"
        }

    # If seismic + not marine → earthquake
    return {
        "detection": "Earthquake",
        "confidence": round(rms_strength, 2),
        "reason": "Sustained low-frequency energy with no biological signature"
    }

# =========================
# RUN
# =========================
if __name__ == "__main__":
    file = "japan.wav"  # change path
    result = detect_event(file)

    print("Final Detection:", result["detection"])
    print("Confidence:", result["confidence"])
    print("Reason:", result["reason"])
