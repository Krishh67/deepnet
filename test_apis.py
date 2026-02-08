"""
Test both API endpoints
"""
import requests
import json

print("=" * 70)
print("🧪 Testing Dual API System")
print("=" * 70)

# Test API 2 (Static Data - Port 8001)
print("\n📊 API 2 (Static Data) - Port 8001")
print("-" * 70)
try:
    # Test root
    r = requests.get("http://localhost:8001/")
    print(f"✅ Root: {r.json()['message']}")
    
    # Test CNN demo
    r = requests.post("http://localhost:8001/predict-seismic-demo", json={"trace_index": 0})
    data = r.json()
    print(f"✅ CNN Demo:")
    print(f"   Detection: {data['detection_type']}")
    print(f"   Confidence: {data['confidence']}%")
    print(f"   Model Accuracy: {data['model_accuracy']}%")
    
    # Test Gemini demo
    r = requests.post("http://localhost:8001/analyze-demo", json={"trace_index": 0})
    data = r.json()
    print(f"✅ Gemini Demo:")
    print(f"   Detection: {data['detection_type']}")
    print(f"   Confidence: {data['confidence']}%")
    print(f"   Tsunami Risk: {data['tsunami_risk']}")
    
except Exception as e:
    print(f"❌ API 2 Error: {e}")

# Test API 1 (Real Models - Port 8000)
print("\n\n🤖 API 1 (Real Models) - Port 8000")
print("-" * 70)
print("⚠️  Note: This will try to load actual CNN + Gemini models")
try:
    # Test root
    r = requests.get("http://localhost:8000/")
    print(f"✅ Root: {r.json().get('message', 'OK')}")
    
    # Test CNN demo (will use real model if loaded)
    r = requests.post("http://localhost:8000/predict-seismic-demo", json={"trace_index": 0}, timeout=10)
    data = r.json()
    print(f"✅ CNN Demo (Real Model):")
    print(f"   Detection: {data['detection_type']}")
    print(f"   Confidence: {data['confidence']}%")
    print(f"   Model Accuracy: {data['model_accuracy']}%")
    
except requests.exceptions.Timeout:
    print(f"⏱️  API 1 timeout - model might be loading")
except Exception as e:
    print(f"❌ API 1 Error: {e}")

print("\n" + "=" * 70)
print("✅ Testing Complete!")
print("=" * 70)
print("\n📝 Summary:")
print("   • API 2 (Port 8001) = Fast static results for demos")
print("   • API 1 (Port 8000) = Real CNN + Gemini models")
print("=" * 70)
