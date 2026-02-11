# 🌊 DeepNet - Hybrid AI for Seismic & Acoustic Event Detection

**Two approaches for two different problems: Deep Learning when we have data, Multimodal AI when we don't**

---

## 📹 Demo Video


https://github.com/user-attachments/assets/f36380b8-deda-4704-9f6d-c033567717b5




<!--
Formats you can use:
- YouTube embed: [![Demo](https://img.youtube.com/vi/VIDEO_ID/maxresdefault.jpg)](https://www.youtube.com/watch?v=VIDEO_ID)
- Local video: ![Demo](./path/to/video.mp4)
- GIF: ![Demo](./demo.gif)
-->

---

## 🎯 What I'm Building

This project solves **two different acoustic classification problems** using the right tool for each:

### Problem 1: Seismic Earthquake Detection (We HAD Data ✅)
We had access to **25,000 labeled seismic waveforms** from the Stanford Earthquake Dataset (STEAD) - a 2.8 GB dataset with ground truth labels including magnitude, P-wave arrivals, and event classifications.

**Solution:** Train a 1D CNN on the 3-channel seismic data
- **Dataset:** STEAD 3C dataset (25,000 traces from UUSS - University of Utah Seismograph Stations)
- **Result:** 97.3% accuracy in earthquake vs noise classification
- **Why it works:** Supervised learning with real seismic patterns from geophysical repositories

### Problem 2: Hydrophone Sound Classification (We DIDN'T Have Data ❌)
For underwater hydrophone recordings (marine life, explosions, ambient noise), we had **no labeled training data**.

**Solution:** Use Gemini's multimodal capabilities
- Extract physics features (RMS energy, duration)
- Generate spectrogram visualization
- Feed both + raw audio to Gemini for zero-shot classification
- **Result:** Explainable classifications without needing thousands of labeled examples


### The Two Approaches

**Approach 1: Supervised Deep Learning (Seismic Events)**
```
Seismic Audio → Preprocessing → 1D CNN → Binary Classification
                (3-channel,      (Trained)  (Earthquake vs Noise)
                 bandpass,
                 normalize)
```

**Approach 2: Multimodal Zero-Shot AI (Hydrophone Sounds)**
```
Hydrophone Audio → Physics Features + Spectrogram + Raw Audio → Gemini → Multi-class Classification
                   (RMS, duration)   (0-50 Hz viz)                       (Earthquake/Explosion/
                                                                           Marine Life/Noise)
```

---

## 🧪 Why This Dual Approach?

### The Data Availability Problem
In real-world ML, you don't always have labeled training data. I had to adapt:

**When I HAD data (Seismic):**
- 25,000 labeled seismic traces from STEAD dataset (2.8 GB)
- Includes magnitude, P-wave arrivals, event classifications
- Perfect for supervised learning
- Trained a custom 1D CNN → 97.3% accuracy

**When I DIDN'T have data (Hydrophone):**
- No labeled examples of whale sounds, explosions, etc.
- Can't train a CNN without data
- Solution: Use Gemini's pre-trained multimodal understanding
- Extract physics features to guide the AI
- Generate spectrograms for visual pattern recognition

### Why Not Just Use Gemini for Everything?
1. **Speed** - CNN inference is ~0.2s vs Gemini's ~3-5s
2. **Cost** - CNN runs locally for free, Gemini costs per API call
3. **Accuracy** - When you have labeled data, supervised learning wins
4. **Reliability** - CNN predictions are deterministic, Gemini can vary

### Why Not Just Use CNN for Everything?
1. **No training data** - Can't train a CNN without labeled examples
2. **No explainability** - CNN gives a number, Gemini explains reasoning
3. **Limited classes** - My CNN does binary classification, Gemini does 4+ classes
4. **Generalization** - CNN trained on seismic data won't work on marine sounds

---

## 🔬 Technical Implementation

### Approach 1: 1D CNN for Seismic Detection

I designed a 4-layer CNN that processes 3-channel seismic data:

```python
Input: (3, 2000) - 3 channels × 2000 samples @ 200Hz
                   Z (Vertical), N (North), E (East) components
├── Conv1D(3→64, k=7) → ReLU → MaxPool → Extract local waveform features
├── Conv1D(64→128, k=5) → ReLU → MaxPool → Detect P-wave onset patterns  
├── Conv1D(128→256, k=3) → ReLU → MaxPool → Learn frequency relationships
├── Conv1D(256→512, k=3) → ReLU → MaxPool → Temporal energy evolution
└── GlobalAvgPool → Dropout(0.5) → FC(512→1) → Sigmoid

Output: P(Earthquake) ∈ [0, 1]
```

**Training Details:**
- **Dataset:** STEAD (Stanford Earthquake Dataset) - 25,000 3-channel seismic waveforms
  - Source: University of Utah Seismograph Stations (UUSS)
  - Size: 2.8 GB (Benchmark, STEAD 3C, MCU embeddings)
  - DOI: [10.57760/sciencedb.10775](https://www.scidb.cn/en/detail?dataSetId=d38d5d20bf9d481eb4fb7b13b9f5a74d)
- **Accuracy:** 97.3% on held-out test set
- **Loss:** Binary Cross-Entropy
- **Optimizer:** Adam
- **Key Insight:** Global pooling makes the model invariant to exact P-wave arrival time

### Approach 2: Gemini Multimodal for Hydrophone Classification
Instead of just sending audio to Gemini, I'm giving it:
1. **Visual data** - Spectrogram (0-50 Hz) showing frequency evolution
2. **Audio data** - Raw waveform for temporal analysis
3. **Physics features** - RMS confidence, duration, energy distribution

This multimodal approach lets Gemini reason like a human analyst would.

---

## 🛠️ Tech Stack

### Backend (Python)
- **FastAPI** - Async API with automatic OpenAPI docs
- **PyTorch** - CNN model training and inference
- **TensorFlow Hub** - YAMNet for marine life detection
- **Librosa** - Spectrogram generation and audio processing
- **SciPy** - Bandpass filtering (Butterworth, order 4)
- **Google Generative AI** - Gemini 2.5 Flash API

### Frontend (TypeScript/React)
- **Next.js 16** - Server components + App Router
- **Tailwind CSS + shadcn/ui** - Modern, accessible UI
- **Recharts** - Real-time waveform visualization
- **React Dropzone** - File upload handling

### Data Pipeline
- **Soundfile** - Audio I/O (Python 3.13 compatible)
- **NumPy** - Signal processing and normalization
- **Pandas** - Demo data metadata management

---

## 🚀 Running the Project

### Backend Setup
```bash
# Create virtual environment
python -m venv .venv
.venv\Scripts\Activate.ps1  # Windows

# Install dependencies
pip install -r requirements.txt

# Add your Gemini API key to .env
echo "GEMINI_API_KEY=your_key_here" > .env

# Run the FastAPI server
python -m uvicorn api1:app --reload --port 8000
```

### Frontend Setup
```bash
# Install dependencies
npm install

# Run Next.js dev server
npm run dev
```

**Access Points:**
- Frontend: http://localhost:3000
- API Docs: http://localhost:8000/docs
- Backend: http://localhost:8000

---

## 📡 API Endpoints

### `/predict-seismic` - CNN Analysis
Upload audio → CNN processes 3-channel waveform → Returns event probability

**Use case:** Fast, accurate seismic/noise binary classification

### `/analyze` - Full Hybrid Analysis
Upload audio → Physics gate → Spectrogram generation → Gemini multimodal → Detailed classification

**Use case:** Comprehensive event classification with tsunami risk assessment

### `/predict-seismic-demo` - CNN Demo Mode
No file upload needed - analyzes pre-loaded seismic traces with known ground truth

**Use case:** Testing and demonstration without real seismic data

### `/analyze-demo` - Gemini Demo Mode
Simulates full Gemini analysis on pre-loaded data

**Use case:** Demo the full system pipeline

---

## 🎯 Key Features

✅ **Dual Analysis Modes**
- Upload real audio files (.wav, .mp3, .m4a)
- Demo mode with 100 pre-loaded seismic traces

✅ **Real-Time Inference**
- Physics analysis: ~0.5s
- CNN inference: ~0.2s (CPU) / ~0.05s (GPU)
- Gemini analysis: ~3-5s (includes spectrogram generation)

✅ **Explainable AI**
- Gemini provides natural language reasoning
- Spectrogram visualization for human verification
- Physics features (RMS, duration) for validation

✅ **Marine Life Detection**
- YAMNet integration for 521 audio classes
- Specifically detects whale, dolphin, seal, and other marine species
- Helps reduce false positives from biological sounds

---

## 📊 Project Structure

```
deepnet/
├── api1.py                    # Main FastAPI application
├── backend/
│   ├── seismic_cnn.pth       # Trained PyTorch model (97% accuracy)
│   ├── demo_data.json        # 100 labeled seismic traces
│   ├── demo_meta.csv         # Metadata (magnitude, P-arrival times)
│   └── live_demo.py          # Standalone inference script
├── components/
│   ├── analyze/              # Gemini analysis UI
│   ├── cnn-analysis/         # CNN analysis UI  
│   └── landing/              # Hero section
├── app/                      # Next.js pages
├── requirements.txt          # Python dependencies
└── package.json              # Node.js dependencies
```

---

## 🧠 What I Learned

### The Key Insight: Match Your Approach to Your Data
The biggest lesson was **you can't force a one-size-fits-all ML solution**:
- Had labeled seismic data → Trained a CNN (supervised learning)
- Didn't have labeled hydrophone data → Used Gemini multimodal (zero-shot learning)

This is real-world ML: adapt to what data you have available.

### Technical Challenges Solved

**1. Python 3.13 Audio Compatibility**
- Problem: `librosa.load()` broke on Python 3.13
- Solution: Switched to `soundfile` for audio I/O + manual resampling

**2. Gemini API Rate Limiting**
- Problem: 429 errors during batch testing
- Solution: Exponential backoff (2s, 4s, 8s delays) with max 3 retries

**3. 3-Channel Simulation from Mono Audio**
- Problem: Seismic sensors have Z/N/E channels, but test audio is mono
- Solution: Applied different bandpass filters to simulate 3-channel data
  ```python
  Z = bandpass(audio, 1, 20, sr)   # Vertical
  N = bandpass(audio, 2, 15, sr)   # North  
  E = bandpass(audio, 3, 18, sr)   # East
  ```

**4. Model Loading Performance**
- Problem: Loading TensorFlow + YAMNet at startup caused 10s+ delays
- Solution: Lazy loading - models load only when first API call needs them

### AI/ML Insights

✅ **When to use supervised learning (CNN):**
- You have labeled training data
- Task is well-defined (binary/multi-class classification)
- Need fast, cheap inference
- Accuracy is priority

✅ **When to use zero-shot AI (Gemini):**
- No labeled training data available
- Need explainability (natural language reasoning)
- Multi-class with overlapping categories
- Can tolerate higher latency/cost

✅ **Physics features help both approaches:**
- For CNN: Preprocessing (bandpass, normalization) improves accuracy
- For Gemini: RMS/duration guide the AI's reasoning

---

## 🔮 Future Directions

- [ ] Train on larger seismic dataset (currently 100 demo traces)
- [ ] Add real-time streaming audio analysis (WebSockets)
- [ ] Implement geographic visualization of detected events
- [ ] Fine-tune Gemini on seismic-specific data
- [ ] Add confidence calibration for CNN predictions
- [ ] Support multi-event detection in long audio files
- [ ] Integrate with real seismic sensor APIs

---

## 🤝 Contributing

This is an active research project! If you're interested in:
- Improving the CNN architecture
- Testing with different seismic datasets
- Enhancing the multimodal prompting strategy
- Building better visualizations

Feel free to open issues or PRs!

---

## 📧 Contact

**Krish Patel**  
GitHub: [@Krishh67](https://github.com/Krishh67)  
Project: [DeepNet](https://github.com/Krishh67/deepnet)

---

## 🙏 Acknowledgments

- **STEAD Dataset** - Stanford Earthquake Dataset (25,000 seismic waveforms)
  - Zhi Geng & team at University of Utah Seismograph Station
  - DOI: 10.57760/sciencedb.10775
- **TensorFlow Hub** for YAMNet pre-trained model
- **Google AI** for Gemini API access
- **PyTorch** community for excellent documentation

---

*Built with curiosity and Python 🐍*
