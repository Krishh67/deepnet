# Backend Integration - Quick Start Guide

## Installation

### 1. Install Python Dependencies

Run this command:
```powershell
pip install --user fastapi uvicorn[standard] python-multipart librosa scipy matplotlib numpy google-generativeai python-dotenv Pillow
```

If you get permission errors, use:
```powershell
pip install --user <packages>
```

### 2. Verify .env File

Make sure your `.env` file contains:
```
GEMINI_API_KEY=your_api_key_here
```

## Running the Backend

### Option 1: Using PowerShell Script (Recommended)
```powershell
.\start-backend.ps1
```

### Option 2: Manual Start
```powershell
python -m uvicorn api:app --reload --port 8000
```

The backend will start on `http://localhost:8000`

## Testing the API

1. **Check if backend is running:**
   - Open browser to: `http://localhost:8000`
  - Should see: `{"status":"ok","message":"OceanGuard Audio Analysis API"}`

2. **View API documentation:**
   - Open: `http://localhost:8000/docs`
   - Interactive Swagger UI for testing endpoints

3. **Test from frontend:**
   - Make sure Next.js dev server is running: `npm run dev` (port 3000)
   - Navigate to: `http://localhost:3000/analyze`
   - Upload an audio file (WAV/MP3)
   - Click "Analyze Audio"

## Troubleshooting

**Backend not starting?**
- Check if Python is installed: `python --version`
- Check if dependencies installed: `pip list | Select-String fastapi`
- Check `.env` file exists with GEMINI_API_KEY

**Frontend can't connect?**
- Make sure both servers are running (port 3000 and 8000)
- Check browser console for CORS errors
- Verify backend URL in audio-analysis-section.tsx: `http://localhost:8000/analyze`

**Analysis fails?**
- Check Gemini API key is valid
- Check audio file format (WAV, MP3, FLAC supported)
- Check backend terminal for error messages
