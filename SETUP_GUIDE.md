# 🎯 Quick Setup Guide - Enhanced Exam Monitoring System

## Step 1: Install New Dependencies

```bash
pip install -r requirements.txt
```

This will install:
- ✅ opencv-python
- ✅ numpy
- ✅ pandas
- ✅ scikit-learn
- ✅ streamlit
- ✅ deepface
- ✅ **ultralytics** (NEW - for device detection)
- ✅ **python-dotenv** (NEW - for config)

**First install will take 5-10 minutes** (downloads YOLOv8 model ~75MB)

---

## Step 2: Run the Application

```bash
streamlit run streamlit_dashboard.py
```

Opens at: http://localhost:8501

---

## Features Summary

| Feature | Status | How it Works |
|---------|--------|------------|
| Electronic Device Detection | ✅ Active | Detects phones, laptops using YOLOv8 |
| Multiple Faces Detection | ✅ Active | Alerts if 2+ people in frame |
| Tab Switching Detection | ✅ Active | Ends exam if student leaves tab |
| Suspicious Activity Tracking | ✅ Active | Counts warnings for violations |
| Auto-End at 5 Warnings | ✅ Active | Terminates exam automatically |

---

## Real-Time Monitoring Display

During exam, you'll see:
- 📹 **Live video feed** with face detection box
- ⏱️ **Time remaining** in seconds
- ⚠️ **Current status** (Normal/Alert)
- 📊 **Warning counter** (X/15) - changes color at 10+
- 😊 **Detected emotion**
- 📈 **Alert timeline** chart

---

## Report Details

After exam, get:
- ✅ Total alerts count
- ✅ Frames analyzed
- ✅ Alert type breakdown (pie chart)
- ✅ Detailed alert table with:
  - Frame number
  - Alert type (device, face, emotion, etc)
  - Severity level
  - Alert message
- ✅ **Download as JSON** for records

---

## Example Workflow

1. **Start**: Click "🚀 Start Calibration"
2. **Calibrate** (60 sec): Sit normally, let system learn baseline
3. **Confirm**: Click "▶️ Start Monitoring"
4. **Monitor** (180 sec default): 
   - If phone detected → ⚠️ Warning +1
   - If 2 people detected → ⚠️ Warning +1
   - If head movement → ⚠️ Warning +1
   - If emotion spike → ⚠️ Warning +1
   - If warnings hit 15 → 🚨 EXAM ENDS
   - If tab switched → 🚨 EXAM ENDS
5. **Report**: Review all violations and download

---

## Customization

To adjust warning limit, edit `streamlit_dashboard.py`:

```python
# Line ~200 in monitor_exam method
if self.session_data['warning_count'] >= 15:  # Change 15 to desired limit
```

To adjust device detection sensitivity:

```python
# Line ~170 in detect_electronic_devices method
if any(device in class_name.lower() for device in device_classes) and confidence > 0.5:
    # Change 0.5 to: 
    # - 0.3 for more sensitive
    # - 0.7 for less sensitive
```

---

## Troubleshooting

**Issue**: "ModuleNotFoundError: No module named 'ultralytics'"
**Solution**: Run `pip install ultralytics`

**Issue**: Slow performance
**Solution**: 
- Close other applications
- Reduce exam duration for testing
- Check camera settings

**Issue**: "No face detected" constantly
**Solution**:
- Ensure good lighting
- Position face in center of camera
- Check camera is not blocked
- Run `python test_camera.py` to verify camera

**Issue**: YOLOv8 model won't download
**Solution**:
- Check internet connection
- Manually download: `python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"`

---

## Files

- `streamlit_dashboard.py` - Main application ✅ ENHANCED
- `exam_monitor_complete.py` - Standalone version ✅ FIXED
- `requirements.txt` - Dependencies ✅ UPDATED
- `test_camera.py` - Camera test utility ✅ CLEAN
- `test_deepface.py` - Face detection test ✅ CLEAN
- `test_install.py` - Package check utility ✅ CLEAN
- `NEW_FEATURES.md` - Detailed feature docs

---

## System Requirements

- ✅ Windows 10+ / Mac / Linux
- ✅ Python 3.8+
- ✅ Webcam
- ✅ 4GB RAM (8GB recommended)
- ✅ 500MB disk space (for models)
- ✅ Stable internet (first run only)

---

## Next Steps

1. ✅ Install dependencies
2. ✅ Run application
3. ✅ Test with calibration (30 seconds)
4. ✅ Try 2-minute exam
5. ✅ Download report
6. ✅ Review logs

Ready to go! 🚀

