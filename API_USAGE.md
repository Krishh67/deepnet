# API Usage Guide

## 🎯 When to Use Which API

### API 1 (Real Models) - Port 8000
**File:** `api1.py`
**Command:** `.venv\Scripts\python -m uvicorn api1:app --reload --port 8000`

**Features:**
- ✅ Real CNN model (`backend/seismic_cnn.pth`)
- ✅ Real Gemini AI integration
- ✅ Actual inference on demo data
- ⚠️ Slower startup (loads models)
- ⚠️ Requires GEMINI_API_KEY in .env

**Use when:**
- Testing actual model performance
- Need real predictions on demo traces
- Validating model accuracy
- Final demo/presentation

---

### API 2 (Static Data) - Port 8001
**File:** `api2.py`
**Command:** `.venv\Scripts\python -m uvicorn api2:app --reload --port 8001`

**Features:**
- ✅ Instant startup (no model loading)
- ✅ Fast static results
- ✅ Perfect for UI/UX testing
- ✅ No API keys needed
- ⚠️ Returns simulated data

**Use when:**
- Developing/testing frontend
- Quick demos without waiting
- UI/UX testing
- Backend is down/slow

---

## 🚀 Quick Start

### Start API 2 (Recommended for testing)
```powershell
.venv\Scripts\python -m uvicorn api2:app --reload --port 8001
```

### Start API 1 (For real models)
```powershell
.venv\Scripts\python -m uvicorn api1:app --reload --port 8000
```

---

## 📡 Endpoints

Both APIs support the same endpoints:

- `GET /` - Health check
- `POST /predict-seismic-demo` - CNN analysis on demo data
- `POST /analyze-demo` - Gemini analysis on demo data
- `POST /predict-seismic` - CNN analysis on uploaded file (API 1 only)
- `POST /analyze` - Gemini analysis on uploaded file (API 1 only)

---

## ⚙️ Frontend Configuration

Update your frontend to point to the desired API:

```typescript
// For static testing (fast)
const API_URL = "http://localhost:8001"

// For real models (slow but accurate)
const API_URL = "http://localhost:8000"
```

---

## ✅ Current Status

- ✅ API 2 running on port 8001 (static data)
- ⏳ API 1 needs testing on port 8000 (real models)
- ✅ CNN model file found: `backend/seismic_cnn.pth`
- ✅ Gemini integration in api1.py
