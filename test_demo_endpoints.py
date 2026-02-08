"""
Quick test script to verify demo endpoints are working
"""
import requests
import json

BASE_URL = "http://localhost:8000"

print("Testing Demo Endpoints...")
print("=" * 60)

# Test 1: Root endpoint
try:
    response = requests.get(f"{BASE_URL}/")
    print(f"✅ Root endpoint: {response.json()}")
except Exception as e:
    print(f"❌ Root endpoint failed: {e}")

# Test 2: Get random demo track
try:
    response = requests.get(f"{BASE_URL}/get-random-demo-track")
    if response.ok:
        data = response.json()
        print(f"\n✅ Random demo track:")
        print(f"   Trace Index: {data['trace_index']}")
        print(f"   Trace ID: {data['trace_id']}")
        print(f"   Magnitude: {data['source_magnitude']}")
        print(f"   P-wave sample: {data['p_arrival_sample']}")
        
        # Test 3: Predict with this trace
        trace_idx = data['trace_index']
        print(f"\n🔍 Testing CNN prediction for trace #{trace_idx}...")
        pred_response = requests.post(
            f"{BASE_URL}/predict-seismic-demo",
            json={"trace_index": trace_idx}
        )
        if pred_response.ok:
            pred_data = pred_response.json()
            print(f"✅ CNN Prediction:")
            print(f"   Detection: {pred_data['detection_type']}")
            print(f"   Confidence: {pred_data['confidence']}%")
            print(f"   Model Accuracy: {pred_data['model_accuracy']}%")
            print(f"   Description: {pred_data['description'][:100]}...")
        else:
            print(f"❌ Prediction failed: {pred_response.text}")
    else:
        print(f"❌ Get random track failed: {response.text}")
except Exception as e:
    print(f"❌ Demo track test failed: {e}")

print("\n" + "=" * 60)
print("Test complete!")
