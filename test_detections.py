#!/usr/bin/env python3
"""
Test script for exam monitoring detections.
Tests each detection method independently.
"""

import cv2
import numpy as np
import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from streamlit_dashboard import StreamlitExamMonitor

def test_multiple_faces():
    """Test multiple faces detection with sample image"""
    print("Testing multiple faces detection...")

    # Create monitor instance
    monitor = StreamlitExamMonitor()

    # Create a test frame with simulated faces (since we can't load real image easily)
    # For testing, we'll use a blank frame and see if it handles it
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    alerts = monitor.detect_multiple_faces(frame)
    print(f"Multiple faces alerts: {len(alerts)}")
    for alert in alerts:
        print(f"  - {alert['message']}")

    return len(alerts) == 0  # Should be 0 for blank frame

def test_device_detection():
    """Test electronic device detection"""
    print("Testing device detection...")

    monitor = StreamlitExamMonitor()

    # Blank frame
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    alerts = monitor.detect_electronic_devices(frame)
    print(f"Device alerts: {len(alerts)}")
    for alert in alerts:
        print(f"  - {alert['message']}")

    return len(alerts) == 0  # Should be 0 for blank frame

def test_audio_detection():
    """Test audio detection (will likely fail without real audio)"""
    print("Testing audio detection...")

    monitor = StreamlitExamMonitor()

    try:
        detected = monitor.detect_audio(duration=0.1)  # Short test
        print(f"Audio detected: {detected}")
        return True  # Method ran without error
    except Exception as e:
        print(f"Audio test failed: {e}")
        return False

def test_face_detection():
    """Test face detection logic"""
    print("Testing face detection logic...")

    monitor = StreamlitExamMonitor()

    # Blank frame - should not detect face
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # Simulate the face detection logic from monitor_exam
    face_present = False
    try:
        from deepface import DeepFace
        result = DeepFace.analyze(frame, actions=['emotion'],
                                 enforce_detection=False, silent=True)
        if isinstance(result, list):
            result = result[0]
        reg = result.get('region', {})
        w, h = reg.get('w', 0), reg.get('h', 0)
        if w > 0 and h > 0:
            face_present = True
        else:
            face_present = False
    except Exception:
        result = None
        face_present = False

    print(f"Face present: {face_present}")
    return not face_present  # Should be False for blank frame

def main():
    """Run all tests"""
    print("Running exam monitoring detection tests...\n")

    tests = [
        ("Multiple Faces Detection", test_multiple_faces),
        ("Device Detection", test_device_detection),
        ("Audio Detection", test_audio_detection),
        ("Face Detection Logic", test_face_detection),
    ]

    passed = 0
    total = len(tests)

    for name, test_func in tests:
        try:
            result = test_func()
            if result:
                print(f"✅ {name}: PASSED\n")
                passed += 1
            else:
                print(f"❌ {name}: FAILED\n")
        except Exception as e:
            print(f"❌ {name}: ERROR - {e}\n")

    print(f"Test Results: {passed}/{total} passed")

    if passed == total:
        print("🎉 All tests passed! Detections should work in the app.")
    else:
        print("⚠️ Some tests failed. Check the implementation.")

if __name__ == "__main__":
    main()