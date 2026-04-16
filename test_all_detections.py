#!/usr/bin/env python3
"""
Comprehensive test for all exam monitoring detections.
This script helps debug why alerts aren't triggering.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from streamlit_dashboard import StreamlitExamMonitor
import numpy as np
import cv2

def test_all_detections():
    """Test each detection method in detail"""
    print("="*60)
    print("EXAM MONITORING DETECTION TEST SUITE")
    print("="*60)
    
    monitor = StreamlitExamMonitor()
    
    # Create test frames
    blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    print("\n1️⃣ TESTING FACE DETECTION LOGIC")
    print("-" * 60)
    try:
        from deepface import DeepFace
        result = DeepFace.analyze(blank_frame, actions=['emotion'], 
                                 enforce_detection=False, silent=True)
        if isinstance(result, list):
            result = result[0]
        reg = result.get('region', {})
        w, h = reg.get('w', 0), reg.get('h', 0)
        face_present = w > 0 and h > 0
        print(f"✓ DeepFace works: face_present={face_present}")
        print(f"  - Region: w={w}, h={h}")
    except Exception as e:
        print(f"✗ DeepFace error: {e}")
    
    print("\n2️⃣ TESTING MULTIPLE FACES DETECTION")
    print("-" * 60)
    try:
        alerts = monitor.detect_multiple_faces(blank_frame)
        print(f"✓ Haar cascade detection works: {len(alerts)} alerts")
        # Check if cascade loaded
        gray = cv2.cvtColor(blank_frame, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(gray, cv2.COLOR_RGB2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
        print(f"  - Cascade loaded: {face_cascade.empty()== False}")
        print(f"  - Detected faces in blank frame: {len(faces)}")
    except Exception as e:
        print(f"✗ Multiple faces error: {e}")
    
    print("\n3️⃣ TESTING DEVICE DETECTION")
    print("-" * 60)
    try:
        alerts = monitor.detect_electronic_devices(blank_frame)
        print(f"✓ YOLO detection works: {len(alerts)} alerts")
        if monitor.device_detector is not None:
            print(f"  - YOLO model loaded: Yes")
        else:
            print(f"  - YOLO model loaded: No (but that's expected)")
    except Exception as e:
        print(f"✗ Device detection error: {e}")
    
    print("\n4️⃣ TESTING AUDIO DETECTION")
    print("-" * 60)
    try:
        import sounddevice as sd
        print(f"✓ Sounddevice available: Yes")
        
        # Test recording
        print(f"  - Testing 0.5s audio recording...")
        detected = monitor.detect_audio(duration=0.5)
        print(f"  - Audio recorded: Yes")
        print(f"  - Detected: {detected}")
        print(f"  - Baseline audio level: {monitor.baseline_audio_level}")
        print(f"  - AUDIO_RMS_THRESHOLD: 0.01")
    except ImportError:
        print(f"✗ Sounddevice NOT installed - this is the problem!")
    except Exception as e:
        print(f"✗ Audio detection error: {e}")
    
    print("\n5️⃣ TESTING CONFIGURATION")
    print("-" * 60)
    print(f"✓ Detection Settings:")
    print(f"  - AUDIO_CHECK_INTERVAL_FRAMES: 30 (check every 1s)")
    print(f"  - AUDIO_RMS_THRESHOLD: 0.01 (lower = more sensitive)")
    print(f"  - Multiple faces threshold: {monitor.multiple_faces_threshold} frames")
    print(f"  - No-face timeout: 7 seconds")
    print(f"  - DEBUG_MODE: Enabled")
    
    print("\n" + "="*60)
    print("TEST COMPLETE")
    print("="*60)
    print("\n📌 IMPORTANT NOTES:")
    print("  • Audio detection requires sounddevice library")
    print("  • Audio is only checked every 30 frames (1 second)")
    print("  • Make sure microphone permissions are granted")
    print("  • Detections should now trigger alerts even without face")
    print("="*60)

if __name__ == "__main__":
    test_all_detections()
