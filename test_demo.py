"""Quick test script to verify the API works before presentation demo."""
import requests
import json
import sys

sys.stdout.reconfigure(encoding="utf-8")

BASE_URL = "http://localhost:5000"

def test_predict(image_path, label):
    with open(image_path, "rb") as f:
        resp = requests.post(f"{BASE_URL}/predict?gradcam_model=both", files={"image": f})
    
    if resp.status_code != 200:
        print(f"  ❌ FAILED: HTTP {resp.status_code}")
        return False
    
    data = resp.json()
    p = data["prediction"]
    sev = data["severity"]
    
    print(f"=== TEST: {label.upper()} ===")
    print(f"  Predicted: {p['class']} ({p['class_vi']})")
    print(f"  Confidence: {p['confidence']}%")
    print(f"  Has Tumor: {p['has_tumor']}")
    print(f"  Severity: {sev['label']} ({sev['level']})")
    print(f"  Recommendation: {data['recommendation'][:60]}...")
    print(f"  Grad-CAM: {list(data['gradcam'].keys())}")
    print(f"  Probabilities:")
    for cn, info in data["probabilities"].items():
        bar = "█" * int(info["score_pct"] / 5)
        print(f"    {info['label_vi']:20s} {info['score_pct']:6.2f}% {bar}")
    
    correct = p["class"] == label
    print(f"  Result: {'✅ CORRECT' if correct else '❌ WRONG'}")
    print()
    return correct

# Health check
print("Checking server health...")
try:
    health = requests.get(f"{BASE_URL}/health").json()
    print(f"  Status: {health['status']}")
    print(f"  Models loaded: {health['models_loaded']}")
    print(f"  Device: {health['device']}")
    print()
except Exception as e:
    print(f"  ❌ Server not running: {e}")
    sys.exit(1)

# Run tests
results = []
tests = [
    ("data/Testing/glioma/Te-glTr_0000.jpg", "glioma"),
    ("data/Testing/notumor/Te-noTr_0000.jpg", "notumor"),
    ("data/Testing/pituitary/Te-piTr_0000.jpg", "pituitary"),
    ("data/Testing/meningioma/Te-meTr_0000.jpg", "meningioma"),
]

for path, label in tests:
    try:
        ok = test_predict(path, label)
        results.append((label, ok))
    except Exception as e:
        print(f"  ❌ Error testing {label}: {e}")
        results.append((label, False))

# Summary
print("=" * 50)
print("  DEMO READINESS CHECK")
print("=" * 50)
all_pass = all(ok for _, ok in results)
for label, ok in results:
    print(f"  {label:15s} {'✅ PASS' if ok else '❌ FAIL'}")
print()
if all_pass:
    print("  🎉 ALL TESTS PASSED — READY FOR DEMO!")
else:
    print("  ⚠️  Some tests failed. Check before presenting.")
print("=" * 50)
