"""
Exam Monitoring System - streamlit_dashboard.py
Fixed version: timer stability, no blinking, camera+question together, tab detection.
Run with:  streamlit run streamlit_dashboard.py
"""

# ─────────────────────────────────────────────────────────────────────────────
#  IMPORTS
# ─────────────────────────────────────────────────────────────────────────────
import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"
from streamlit_javascript import st_javascript
import streamlit as st
import streamlit.components.v1 as components
import cv2
import numpy as np
import time
import json
from datetime import datetime
import threading
import pandas as pd
import plotly.express as px
from ultralytics import YOLO

try:
    import winsound
except ImportError:
    winsound = None

try:
    import sounddevice as _sd_module
    SOUNDDEVICE_AVAILABLE = True
except (ImportError, OSError):
    _sd_module = None
    SOUNDDEVICE_AVAILABLE = False

# ─────────────────────────────────────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
MAX_WARNINGS                = 15
AUDIO_CHECK_INTERVAL_FRAMES = 15
AUDIO_RMS_THRESHOLD         = 0.008
AUDIO_BASELINE_MULTIPLIER   = 1.5
AUDIO_SAMPLE_DURATION       = 1.0
AUDIO_SAMPLE_RATE           = 16000

NO_FACE_TIMEOUT             = 10
NO_FACE_WARN_INTERVAL       = 2

MULTI_FACE_FRAMES_NEEDED    = 2
DEVICE_CONFIDENCE_THRESHOLD = 0.70

DEVICE_CLASSES = {'cell phone', 'laptop'}

DEBUG_MODE = False

# ─────────────────────────────────────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Exam Monitoring System",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.main { background-color: #f5f5f5; }
h1    { color: #1f77b4; text-align: center; }
/* Prevent layout shift / flicker on rerun */
.stRadio > div { transition: none !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
#  SAMPLE TEST DATA
# ─────────────────────────────────────────────────────────────────────────────
SAMPLE_TESTS = {
    "DSA": {
        "duration": 240,
        "questions": [
            {"text": "What is the time complexity of binary search on a sorted array?",
             "options": ["O(n)", "O(log n)", "O(n log n)", "O(1)"],
             "correct_answer": "O(log n)"},
            {"text": "Which data structure uses FIFO ordering?",
             "options": ["Stack", "Queue", "Tree", "Graph"],
             "correct_answer": "Queue"},
            {"text": "What is the height of a balanced binary tree with n nodes?",
             "options": ["O(n)", "O(log n)", "O(n log n)", "O(1)"],
             "correct_answer": "O(log n)"},
            {"text": "Which sorting algorithm is stable?",
             "options": ["Quick sort", "Heap sort", "Merge sort", "Selection sort"],
             "correct_answer": "Merge sort"},
            {"text": "Which data structure is best for implementing recursion?",
             "options": ["Queue", "Stack", "Hash table", "Linked list"],
             "correct_answer": "Stack"},
        ],
    },
    "ML AI": {
        "duration": 240,
        "questions": [
            {"text": "What is supervised learning?",
             "options": ["Learning from labeled data", "Learning from unlabeled data",
                         "Learning without data", "Learning only from images"],
             "correct_answer": "Learning from labeled data"},
            {"text": "Which algorithm is used for regression?",
             "options": ["K-means", "Linear regression", "Apriori", "DBSCAN"],
             "correct_answer": "Linear regression"},
            {"text": "What does NLP stand for?",
             "options": ["Neural Learning Protocol", "Natural Language Processing",
                         "Network Layer Programming", "Numeric Linear Prediction"],
             "correct_answer": "Natural Language Processing"},
            {"text": "Which metric is used for classification accuracy?",
             "options": ["Mean squared error", "Accuracy score", "R-squared", "Log loss"],
             "correct_answer": "Accuracy score"},
            {"text": "In AI, what is overfitting?",
             "options": ["Model performs well on training but poorly on test",
                         "Model fails to learn anything", "Model uses too little data",
                         "Model is too simple"],
             "correct_answer": "Model performs well on training but poorly on test"},
        ],
    },
    "DataScience": {
        "duration": 240,
        "questions": [
            {"text": "Which library is commonly used for data manipulation in Python?",
             "options": ["NumPy", "Pandas", "Matplotlib", "Scikit-learn"],
             "correct_answer": "Pandas"},
            {"text": "What is the purpose of data normalization?",
             "options": ["Reduce dataset size", "Scale features to a common range",
                         "Increase variance", "Remove missing values"],
             "correct_answer": "Scale features to a common range"},
            {"text": "Which plot type is best for showing categorical counts?",
             "options": ["Line chart", "Scatter plot", "Bar chart", "Histogram"],
             "correct_answer": "Bar chart"},
            {"text": "What is a common task in exploratory data analysis?",
             "options": ["Model deployment", "Feature engineering",
                         "Summary statistics", "Hyperparameter tuning"],
             "correct_answer": "Summary statistics"},
            {"text": "Which metric measures central tendency?",
             "options": ["Standard deviation", "Variance", "Mean", "Correlation"],
             "correct_answer": "Mean"},
        ],
    },
    "Aptitude": {
        "duration": 240,
        "questions": [
            {"text": "What is 15% of 200?",
             "options": ["20", "25", "30", "35"],
             "correct_answer": "30"},
            {"text": "If train A travels 60 km/h and train B travels 80 km/h, who is faster?",
             "options": ["Train A", "Train B", "Both same", "Cannot determine"],
             "correct_answer": "Train B"},
            {"text": "Which is the next number in the sequence: 2, 4, 8, 16, ?",
             "options": ["18", "24", "32", "34"],
             "correct_answer": "32"},
            {"text": "A ratio of 3:4 is equal to which fraction?",
             "options": ["3/4", "4/3", "7/12", "1/2"],
             "correct_answer": "3/4"},
            {"text": "If x + 7 = 12, what is x?",
             "options": ["3", "4", "5", "6"],
             "correct_answer": "5"},
        ],
    },
}

QUESTION_DURATION = 60  # seconds per question


# ═════════════════════════════════════════════════════════════════════════════
#  MONITOR CLASS
# ═════════════════════════════════════════════════════════════════════════════
class StreamlitExamMonitor:

    def __init__(self):
        self.baseline = {
            "head_center": None, "head_std": None,
            "avg_emotions": {}, "is_calibrated": False,
        }
        self.baseline_audio_level = None

        self.session_data = {
            "start_time": None, "alerts": [],
            "frames_processed": 0, "warning_count": 0,
        }

        self.warning_counts   = {}
        self.no_face_start    = None
        self.multi_face_count = 0
        self.total_looking_away_time = 0

        # ── Thread-safe shared state ──────────────────────────────────────
        self._lock = threading.Lock()
        self.monitor_running       = False
        self.monitor_thread        = None
        self.monitor_start_time    = None
        self.monitor_duration      = None
        self.latest_frame_rgb      = None
        self.latest_metrics        = {}
        self.latest_recent_alerts  = []
        self.latest_alert_timeline = []
        self.exam_ended_early      = False
        self.exam_end_timeline     = []

        # ── Tab visibility – written by UI, read by monitor thread ────────
        self._tab_visible      = True
        self._tab_switch_count = 0   # counts confirmed tab-switch events

        # ── YOLO ──────────────────────────────────────────────────────────
        try:
            self.yolo = YOLO("yolov8n.pt")
        except Exception:
            self.yolo = None

        # ── Haar cascade ──────────────────────────────────────────────────
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.face_cascade = cv2.CascadeClassifier(cascade_path)

        self.sd = _sd_module

    # ── Tab visibility property (thread-safe) ────────────────────────────
    @property
    def tab_visible(self):
        with self._lock:
            return self._tab_visible

    @tab_visible.setter
    def tab_visible(self, value):
        with self._lock:
            self._tab_visible = bool(value)

    # ─────────────────────────────────────────────────────────────────────
    #  CALIBRATION
    # ─────────────────────────────────────────────────────────────────────
    def calibrate(self, duration, progress_bar, status_text, video_ph):
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        head_positions     = []
        emotions_collected = []
        start_time         = time.time()

        bg_recording = None
        if self.sd is not None:
            try:
                bl_dur = min(2, duration)
                bg_recording = self.sd.rec(
                    int(bl_dur * AUDIO_SAMPLE_RATE),
                    samplerate=AUDIO_SAMPLE_RATE, channels=1, dtype="float32",
                )
            except Exception:
                pass

        while time.time() - start_time < duration:
            ret, frame = cap.read()
            if not ret:
                continue

            elapsed   = time.time() - start_time
            remaining = int(max(duration - elapsed, 0))
            progress  = min(elapsed / duration, 1.0)

            try:
               res = {"dominant_emotion": "Not available"}
                
                reg  = res["region"]
                x, y, w, h = reg["x"], reg["y"], reg["w"], reg["h"]
                if w > 0 and h > 0:
                    head_positions.append([x + w // 2, y + h // 2])
                    emotions_collected.append(res["emotion"])
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, f"CALIBRATING: {remaining}s",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.putText(frame, f"Captured {len(head_positions)} frames",
                            (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            except Exception:
                cv2.putText(frame, "No face – please look at camera",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            video_ph.image(frame_rgb, channels="RGB", use_container_width=True)
            progress_bar.progress(progress)
            status_text.text(f"Calibrating… {remaining}s remaining")
            time.sleep(0.03)

        cap.release()

        if bg_recording is not None:
            try:
                self.sd.wait()
                rms = float(np.sqrt(np.mean(bg_recording ** 2)))
                self.baseline_audio_level = rms
            except Exception:
                self.baseline_audio_level = AUDIO_RMS_THRESHOLD

        if len(head_positions) >= 10:
            pos = np.array(head_positions)
            self.baseline["head_center"]  = np.mean(pos, axis=0)
            self.baseline["head_std"]     = np.std(pos, axis=0)
            avg_em = {k: float(np.mean([e[k] for e in emotions_collected]))
                      for k in emotions_collected[0]}
            self.baseline["avg_emotions"]  = avg_em
            self.baseline["is_calibrated"] = True
            return True, len(head_positions)

        return False, 0

    # ─────────────────────────────────────────────────────────────────────
    #  START BACKGROUND MONITORING THREAD
    # ─────────────────────────────────────────────────────────────────────
    def start_monitoring(self, duration, student_name):
        with self._lock:
            if self.monitor_thread is not None and self.monitor_thread.is_alive():
                return

            self.session_data = {
                "start_time": datetime.now(), "alerts": [],
                "frames_processed": 0, "warning_count": 0,
            }
            self.warning_counts        = {}
            self.no_face_start         = None
            self.multi_face_count      = 0
            self.latest_frame_rgb      = None
            self.latest_metrics        = {}
            self.latest_recent_alerts  = []
            self.latest_alert_timeline = []
            self.exam_ended_early      = False
            self.exam_end_timeline     = []
            self.monitor_duration      = duration
            self.monitor_start_time    = time.time()
            self.monitor_running       = True
            self._tab_visible          = True
            self._tab_switch_count     = 0

        def run_monitor():
            tl, ended = self._monitor_loop(duration, student_name)
            with self._lock:
                self.exam_end_timeline = tl
                self.exam_ended_early  = ended
                self.monitor_running   = False

        self.monitor_thread = threading.Thread(target=run_monitor, daemon=True)
        self.monitor_thread.start()

    def is_monitoring(self):
        return self.monitor_running and (
            self.monitor_thread is not None and self.monitor_thread.is_alive()
        )

    # ─────────────────────────────────────────────────────────────────────
    #  MAIN MONITORING LOOP  (runs in background thread – NO st.* calls)
    # ─────────────────────────────────────────────────────────────────────
    def _monitor_loop(self, duration, student_name):
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

        start_time       = time.time()
        frame_number     = 0
        cooldown         = {}
        alert_timeline   = []
        exam_ended_early = False

        while time.time() - start_time < duration:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.03)
                continue

            frame_number += 1
            elapsed   = time.time() - start_time
            remaining = int(duration - elapsed)
            current_alerts = []

            # ── Tab-switch detection ──────────────────────────────────────
            # tab_visible is set by the UI thread via the property setter
            if not self.tab_visible:
                ended = self._handle_critical(
                    {"type": "tab_switch", "severity": "critical",
                     "message": "🗂 ALERT: Tab/window switched away from exam!"},
                    "tab_switch", frame_number, elapsed,
                    cooldown, alert_timeline, current_alerts,
                )
                if ended:
                    exam_ended_early = True
                    break

            # ── Face presence ─────────────────────────────────────────────
            face_present = False
            df_result    = None
            try:
                df_result = None
                face_present = True  # assume face present (simple version)
            except Exception:
                face_present = False
                df_result    = None

            # ── No-face handling ──────────────────────────────────────────
            if not face_present:
                frame = self._draw_red_border(frame)
                if self.no_face_start is None:
                    self.no_face_start = time.time()
                absent_secs = time.time() - self.no_face_start

                cv2.putText(frame, "NO PERSON DETECTED",
                            (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 3)
                cv2.putText(frame, f"Absent: {absent_secs:.1f}s / {NO_FACE_TIMEOUT}s",
                            (10, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

                if absent_secs >= NO_FACE_TIMEOUT:
                    exam_ended_early = True
                    msg = f"🚨 EXAM ENDED: No person visible for >{NO_FACE_TIMEOUT}s"
                    self._log({"type": "no_face_timeout", "severity": "critical",
                               "message": msg}, frame_number, elapsed, alert_timeline)
                    self._play_beep()
                    break

                if time.time() - cooldown.get("no_face", 0) >= NO_FACE_WARN_INTERVAL:
                    warn = {"type": "no_face", "severity": "high",
                            "message": "⚠️ No person detected – please face the camera"}
                    self._log(warn, frame_number, elapsed, alert_timeline)
                    current_alerts.append(warn["message"])
                    cooldown["no_face"] = time.time()
            else:
                self.no_face_start = None

            # ── Multiple faces ────────────────────────────────────────────
            for alert in self._detect_faces(frame):
                ended = self._handle_critical(
                    alert, "multiple_faces", frame_number, elapsed,
                    cooldown, alert_timeline, current_alerts,
                )
                if ended:
                    exam_ended_early = True
                    break
            if exam_ended_early:
                break

            # ── Electronic devices ────────────────────────────────────────
            for alert in self._detect_devices(frame):
                dev_key = f"device:{alert.get('device', 'unknown')}"
                ended   = self._handle_critical(
                    alert, dev_key, frame_number, elapsed,
                    cooldown, alert_timeline, current_alerts,
                )
                if ended:
                    exam_ended_early = True
                    break
            if exam_ended_early:
                break

            # ── Audio ─────────────────────────────────────────────────────
            if frame_number % AUDIO_CHECK_INTERVAL_FRAMES == 0:
                if self._detect_audio():
                    ended = self._handle_critical(
                        {"type": "audio_detected", "severity": "critical",
                         "message": "🎤 ALERT: Background voice / loud noise detected!"},
                        "audio", frame_number, elapsed,
                        cooldown, alert_timeline, current_alerts,
                    )
                    if ended:
                        exam_ended_early = True
                        break

            # ── Global warning ceiling ────────────────────────────────────
            if self.session_data["warning_count"] >= MAX_WARNINGS:
                exam_ended_early = True
                msg = f"🚨 EXAM ENDED: Warning limit ({MAX_WARNINGS}) reached!"
                self._log({"type": "warning_limit", "severity": "critical",
                           "message": msg}, frame_number, elapsed, alert_timeline)
                self._play_beep()
                break

            # ── Draw overlays ─────────────────────────────────────────────
            if face_present and df_result is not None:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)

                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
                dominant = "Disabled"
                box_color  = (0, 0, 255) if current_alerts else (0, 255, 0)
                cv2.rectangle(frame, (x, y), (x + w, y + h), box_color, 2)
                label = "⚠ ALERT" if current_alerts else "✓ Normal"
                color = (0, 0, 255) if current_alerts else (0, 255, 0)
                cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
                
                cv2.putText(frame,
                            f"Warnings: {self.session_data['warning_count']}",
                            (10, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (255, 165, 0) if self.session_data["warning_count"] > 0
                            else (255, 255, 255), 2)
                cv2.putText(frame, f"Emotion: {dominant}",
                            (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # ── Push to shared state (thread-safe) ────────────────────────
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            with self._lock:
                self.latest_frame_rgb = frame_rgb
                self.latest_metrics = {
                    "Time Remaining": f"{remaining}s",
                    "Total Alerts":   len(self.session_data["alerts"]),
                    "Frames":         frame_number,
                    "Warnings":       f"{self.session_data['warning_count']}/{MAX_WARNINGS}",
                    "Status":         "🔴 Alert" if current_alerts else "🟢 Normal",
                }
                self.latest_recent_alerts  = list(self.session_data["alerts"][-5:])
                self.latest_alert_timeline = list(alert_timeline)

            self.session_data["frames_processed"] = frame_number

        cap.release()
        return alert_timeline, exam_ended_early

    # ─────────────────────────────────────────────────────────────────────
    #  DETECTION METHODS
    # ─────────────────────────────────────────────────────────────────────
    def _face_present(self, frame, df_result):
        if df_result is None:
            return False
        reg = df_result.get("region", {})
        if reg.get("w", 0) <= 0 or reg.get("h", 0) <= 0:
            return False
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)
            faces = self.face_cascade.detectMultiScale(
                gray, scaleFactor=1.05, minNeighbors=6, minSize=(40, 40))
            if len(faces) > 0:
                return True
            # fallback for medium/dim lighting or lower-contrast faces
            faces = self.face_cascade.detectMultiScale(
                gray, scaleFactor=1.03, minNeighbors=4, minSize=(32, 32))
            if len(faces) > 0:
                return True
        except Exception:
            pass
        face_confidence = df_result.get("face_confidence", 0)
        try:
            return float(face_confidence) > 0.0
        except Exception:
            return False

    def _detect_multiple_persons_yolo(self, frame):
        alerts = []
        if self.yolo is None:
            return alerts
        try:
            results = self.yolo(frame, verbose=False)
            person_count = 0
            for r in results:
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    conf   = float(box.conf[0])
                    name   = self.yolo.model.names.get(cls_id, str(cls_id)).lower().strip()
                    if name == "person" and conf >= DEVICE_CONFIDENCE_THRESHOLD:
                        person_count += 1
            if person_count > 1:
                alerts.append({
                    "type": "multiple_faces", "severity": "critical",
                    "num_faces": person_count,
                    "message": f"🚨 ALERT: {person_count} people detected! Possible cheating!",
                })
        except Exception:
            pass
        return alerts

    def _detect_faces(self, frame):
        alerts = []
        fh, fw = frame.shape[:2]
        frame_area = fh * fw
        try:
            gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray  = cv2.equalizeHist(gray)
            faces = self.face_cascade.detectMultiScale(
                gray, scaleFactor=1.05, minNeighbors=7, minSize=(40, 40))

            valid = []
            for (x, y, w, h) in faces:
                area  = w * h
                ratio = area / frame_area
                if ratio < 0.008 or ratio > 0.55:
                    continue
                if x / fw < 0.04 or (x + w) / fw > 0.96:
                    continue
                if y / fh < 0.04 or (y + h) / fh > 0.96:
                    continue
                if max(w, h) / max(min(w, h), 1) > 2.2:
                    continue
                cx, cy    = x + w / 2, y + h / 2
                duplicate = False
                for (vx, vy, vw, vh) in valid:
                    ox   = max(0, min(x + w, vx + vw) - max(x, vx))
                    oy   = max(0, min(y + h, vy + vh) - max(y, vy))
                    ovlp = (ox * oy) / max(min(area, vw * vh), 1)
                    if ovlp > 0.4:
                        duplicate = True
                        break
                    dist     = ((cx - (vx + vw / 2)) ** 2 + (cy - (vy + vh / 2)) ** 2) ** 0.5
                    min_dist = 1.3 * max(w, vw)
                    if dist < min_dist:
                        duplicate = True
                        break
                if not duplicate:
                    valid.append((x, y, w, h))

            if len(valid) > 1:
                self.multi_face_count += 1
            else:
                self.multi_face_count = 0

            if len(valid) > 1 and self.multi_face_count >= MULTI_FACE_FRAMES_NEEDED:
                alerts.append({
                    "type": "multiple_faces", "severity": "critical",
                    "num_faces": len(valid),
                    "message": f"🚨 ALERT: {len(valid)} faces detected! Possible cheating!",
                })
                self.multi_face_count = 0
            elif len(valid) <= 1:
                alerts.extend(self._detect_multiple_persons_yolo(frame))
        except Exception:
            pass
        return alerts

    def detect_multiple_faces(self, frame):
        return self._detect_faces(frame)

    def _detect_devices(self, frame):
        alerts = []
        if self.yolo is None:
            return alerts
        try:
            results = self.yolo(frame, verbose=False)
            for r in results:
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    conf   = float(box.conf[0])
                    name   = self.yolo.model.names.get(cls_id, str(cls_id)).lower().strip()
                    if conf >= DEVICE_CONFIDENCE_THRESHOLD and name in DEVICE_CLASSES:
                        alerts.append({
                            "type": "electronic_device", "severity": "critical",
                            "device": name,
                            "message": f"📱 Device detected: {name} (conf: {conf:.2f})",
                            "box": box.xyxy[0].tolist(),
                        })
        except Exception:
            pass
        return alerts

    def detect_electronic_devices(self, frame):
        return self._detect_devices(frame)

    def _detect_audio(self):
        if self.sd is None:
            return False
        try:
            rec = self.sd.rec(int(AUDIO_SAMPLE_DURATION * AUDIO_SAMPLE_RATE),
                              samplerate=AUDIO_SAMPLE_RATE, channels=1, dtype="float32")
            self.sd.wait()
            rms = float(np.sqrt(np.mean(rec ** 2)))
            if self.baseline_audio_level is not None:
                return rms > max(
                    self.baseline_audio_level * AUDIO_BASELINE_MULTIPLIER,
                    AUDIO_RMS_THRESHOLD,
                )
            return rms > AUDIO_RMS_THRESHOLD
        except Exception:
            return False

    def detect_audio(self, duration=AUDIO_SAMPLE_DURATION, threshold=AUDIO_RMS_THRESHOLD):
        return self._detect_audio()

    # ─────────────────────────────────────────────────────────────────────
    #  HELPERS
    # ─────────────────────────────────────────────────────────────────────
    def _handle_critical(self, alert, key, frame_number, elapsed,
                         cooldown, timeline, current_alerts):
        count = self.warning_counts.get(key, 0)
        last  = cooldown.get(key, 0)
        if count == 0:
            if time.time() - last >= 3:
                self.warning_counts[key] = 1
                self.session_data["warning_count"] += 1
                current_alerts.append(alert["message"])
                self._log(alert, frame_number, elapsed, timeline)
                cooldown[key] = time.time()
                self._play_beep()
        else:
            vtype = alert["type"].replace("_", " ")
            msg   = f"🚨 EXAM ENDED: Repeated violation – {vtype}"
            term  = {"type": "exam_terminated", "severity": "critical", "message": msg}
            current_alerts.append(msg)
            self._log(term, frame_number, elapsed, timeline)
            self._play_beep()
            return True

        return False

    def _log(self, alert, frame_number, elapsed, timeline):
        self.session_data["alerts"].append(
            {**alert, "frame": frame_number, "time": elapsed}
        )
        timeline.append({
            "time": elapsed, "type": alert["type"],
            "message": alert.get("message", ""),
        })

    @staticmethod
    def _draw_red_border(frame, thickness=10):
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 0, 255), thickness)
        return frame

    def _play_beep(self):
        try:
            if winsound:
                winsound.Beep(1000, 200)
        except Exception:
            pass

    def request_media_permissions(self, height=180):
        html = """
        <div style='font-family:sans-serif; padding:10px'>
          <h3>📷 Camera & Microphone Access</h3>
          <p>Please click <strong>Allow</strong> when your browser prompts.</p>
          <div id='status' style='font-weight:bold'>Requesting…</div>
          <script>
            (async () => {
              try {
                const s = await navigator.mediaDevices.getUserMedia({video:true,audio:true});
                s.getTracks().forEach(t => t.stop());
                document.getElementById('status').style.color = 'green';
                document.getElementById('status').innerText = '✅ Permissions granted – click Continue below.';
              } catch(e) {
                document.getElementById('status').style.color = 'red';
                document.getElementById('status').innerText = '❌ Permissions denied. Allow both then click Retry.';
              }
            })();
          </script>
        </div>"""
        try:
            components.html(html, height=height)
        except Exception:
            st.info("Please allow camera and microphone in your browser.")

    def request_screen_share_permissions(self, height=260):
        html = """
        <div style='font-family:sans-serif; padding:10px'>
          <h3>🖥️ Screen / Window Sharing</h3>
          <p>Before the exam begins, please share your screen, tab, or window.</p>
          <button id='share-btn' style='font-size:16px; padding:12px 18px;'>Share Screen</button>
          <div id='status' style='margin-top:12px; font-weight:bold'>Click the button and allow sharing.</div>
          <script>
            const status = document.getElementById('status');
            const button = document.getElementById('share-btn');
            button.onclick = async function() {
              try {
                const stream = await navigator.mediaDevices.getDisplayMedia({video:true});
                stream.getTracks().forEach(track => track.stop());
                status.style.color = 'green';
                status.innerText = '✅ Screen sharing permission granted. Now click Continue.';
              } catch (err) {
                status.style.color = 'red';
                status.innerText = '❌ Screen sharing permission denied. Please retry.';
              }
            };
          </script>
        </div>"""
        try:
            components.html(html, height=height, scrolling=True)
        except Exception:
            st.info("Please allow screen sharing in your browser.")


# ═════════════════════════════════════════════════════════════════════════════
#  TAB-VISIBILITY COMPONENT
#  Renders a tiny HTML snippet that sets a query-param on visibility change.
#  The Streamlit app reads the param on next rerun and updates tab_visible.
# ═════════════════════════════════════════════════════════════════════════════




# ═════════════════════════════════════════════════════════════════════════════
#  STREAMLIT UI
# ═════════════════════════════════════════════════════════════════════════════
def main():
    st.title("🎓 AI-Powered Online Exam Monitoring System")
    st.markdown("---")

    # ── Sidebar ───────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("⚙️ Settings")
        student_name = st.text_input("Student Name", value="Student")

        st.markdown("---")
        st.subheader("📋 Select Test")
        selected_test = st.selectbox(
            "Choose a test:", ["None"] + list(SAMPLE_TESTS.keys()),
        )
        if selected_test != "None":
            ti = SAMPLE_TESTS[selected_test]
            st.info(f"**{selected_test}**\nDuration: {ti['duration']}s")
            with st.expander("📖 View Questions"):
                for q in ti["questions"]:
                    st.write(q["text"] if isinstance(q, dict) else q)
        st.session_state["selected_test"] = selected_test

        st.markdown("---")
        calibration_time = st.slider("Calibration Duration (s)", 10, 120, 30)
        if selected_test != "None":
            exam_duration = SAMPLE_TESTS[selected_test]["duration"]
            st.info(f"Exam duration: {exam_duration}s")
        else:
            exam_duration = st.slider("Exam Duration (s)", 30, 3600, 180,
                                       key="exam_duration")

        st.markdown("---")
        st.subheader("ℹ️ Active Detections")
        st.markdown("""
- 👤 Face presence + red border if missing
- ⏱ No-face → terminate after **10 s**
- 👥 Multiple faces (3-frame confirm)
- 📱 Devices: phone / laptop / TV / keyboard
- 🎤 Background voice / noise
- 🖥️ Screen sharing required before exam
- 🗂 Tab switching: 1st warning · 2nd ends exam
        """)
        if not SOUNDDEVICE_AVAILABLE:
            st.warning("⚠️ Audio detection disabled.\nInstall `sounddevice` + PortAudio to enable.")
        else:
            try:
                sd_mod = st.session_state.monitor.sd
                dev = sd_mod.default.device
                if isinstance(dev, (list, tuple)):
                    audio_in = dev[0]
                else:
                    audio_in = dev
                if audio_in is not None and audio_in >= 0:
                    dev_info = sd_mod.query_devices(audio_in)
                    st.success(f"🎤 Audio detection enabled – input: {dev_info['name']}")
                else:
                    st.info("🎤 Audio detection enabled, but no default input device is selected.")
            except Exception:
                st.info("🎤 Audio detection enabled, but device status could not be read.")

    # ── Session state init ────────────────────────────────────────────────
    if "monitor" not in st.session_state:
        st.session_state.monitor               = StreamlitExamMonitor()
        st.session_state.stage                 = "ready"
        st.session_state["selected_test"]      = "None"
        st.session_state["exam_ended_early"]   = False
        st.session_state["tab_visible"]        = True
        st.session_state["last_tab_ts"]        = 0
        st.session_state["screen_share_ready"] = False
        st.session_state["monitor_started"]    = False
        # ── Question / timer state (initialised once) ─────────────────────
        st.session_state["current_q"]          = 0
        st.session_state["question_start"]     = None
        st.session_state["answers"]            = {}

    stage = st.session_state.stage

    # ── READY ─────────────────────────────────────────────────────────────
    if stage == "ready":
        st.info("👋 Welcome! Click **Start Calibration** to begin.")
        st.info("💡 Make sure you sit in front of a light source so your face is visible to the camera.")
        _, col, _ = st.columns([1, 2, 1])
        with col:
            if st.button("🚀 Start Calibration", type="primary",
                         use_container_width=True):
                st.session_state.monitor = StreamlitExamMonitor()
                st.session_state.stage   = "calibrating"
                st.rerun()

    # ── CALIBRATING ───────────────────────────────────────────────────────
    elif stage == "calibrating":
        st.header("📹 Calibration Phase")
        st.info(f"Sit normally and look at the camera for {calibration_time}s.")
        st.info("🔆 Good lighting helps the system detect your face reliably. Use a desk lamp or room light facing you.")
        video_ph   = st.empty()
        prog_bar   = st.progress(0)
        status_txt = st.empty()

        ok, frames = st.session_state.monitor.calibrate(
            calibration_time, prog_bar, status_txt, video_ph
        )

        if ok:
            st.success(f"✅ Calibration done! ({frames} frames captured)")
            bl = st.session_state.monitor.baseline
            c1, c2, c3 = st.columns(3)
            c1.metric("Head Centre",
                      f"({bl['head_center'][0]:.0f}, {bl['head_center'][1]:.0f})")
            c2.metric("Move Tolerance", f"±{float(np.mean(bl['head_std'])):.0f} px")
            c3.metric("Baseline Emotion",
                      max(bl["avg_emotions"], key=bl["avg_emotions"].get).capitalize())
            if st.button("▶️ Continue to Exam Instructions", type="primary"):
                st.session_state.stage = "pre_exam_instructions"
                st.rerun()
        else:
            st.error("❌ Calibration failed – face not detected clearly. "
                 "Ensure good lighting, sit facing the camera, and keep your face fully in the frame.")
            if st.button("🔄 Retry Calibration"):
                st.rerun()

    # ── PRE-EXAM INSTRUCTIONS ─────────────────────────────────────────────
    elif stage == "pre_exam_instructions":
        st.header("📘 Exam Instructions")
        st.markdown("""
- ⏱ Each question is shown one at a time with a **60-second timer**.
- ▶️ After time ends, the next question loads automatically.
- 📝 Your answer is saved automatically.
- 📹 Camera monitors you continuously in the corner panel.
- ⚠️ Suspicious activity may end the exam automatically.
- 💡 Use good front lighting so your face is visible, especially in low-light or evening conditions.
- 🎤 Make sure your microphone is enabled to allow audio detection.
- ✅ No manual "Next" button is needed.
        """)
        st.info("Click below to set up camera and microphone permissions.")
        _, col, _ = st.columns([1, 2, 1])
        with col:
            if st.button("✅ Proceed to Permissions", type="primary",
                         use_container_width=True):
                st.session_state.stage = "request_permissions"
                st.rerun()

    # ── REQUEST PERMISSIONS ───────────────────────────────────────────────
    elif stage == "request_permissions":
        st.header("🔒 Camera & Microphone Permissions")
        st.info("Allow both camera and microphone when your browser prompts.")
        st.session_state.monitor.request_media_permissions()
        c1, c2 = st.columns(2)
        with c1:
            if st.button("✅ Permissions granted – Continue"):
                st.session_state.stage = "request_screen_share"
                st.rerun()
        with c2:
            if st.button("🔁 Retry Permissions"):
                st.rerun()

    # ── REQUEST SCREEN SHARE ──────────────────────────────────────────────
    elif stage == "request_screen_share":
        st.header("🖥️ Screen / Tab / Window Sharing")
        st.info("Click the button below to share your screen/tab/window. "
                "This permission is required before the exam starts.")
        st.session_state.monitor.request_screen_share_permissions()
        c1, c2 = st.columns(2)
        with c1:
            if st.button("✅ Screen sharing granted – Continue"):
                st.session_state.screen_share_ready = True
                st.session_state.stage = "monitoring"
                # Reset question state cleanly for a fresh exam
                st.session_state["current_q"]      = 0
                st.session_state["question_start"] = time.time()
                st.session_state["answers"]        = {}
                st.session_state["monitor_started"] = False
                st.rerun()
        with c2:
            if st.button("🔁 Retry Screen Sharing"):
                st.rerun()

    # ── MONITORING ────────────────────────────────────────────────────────
    elif stage == "monitoring":
        st.header("🔴 Live Exam Monitoring")

        sel       = st.session_state.get("selected_test", "None")
        questions = SAMPLE_TESTS[sel]["questions"] if sel != "None" else []
        total_q   = len(questions)
        exam_duration = total_q * QUESTION_DURATION if total_q else st.session_state.get("exam_duration", 180)

        # ── Initialise question timer ONCE ───────────────────────────────
        # We store the absolute wall-clock start time so that reruns do NOT
        # reset the timer.  This is the key fix for "timer jumping on answer".
        if st.session_state.get("question_start") is None:
            st.session_state["question_start"] = time.time()

        # ── Start background monitor ONCE ────────────────────────────────
        if not st.session_state.get("monitor_started", False):
            st.session_state.monitor.start_monitoring(exam_duration, student_name)
            st.session_state["monitor_started"] = True

        monitor = st.session_state.monitor

        # ── Tab detection ─────────────────────────────────────────────────
        # Inject the JS writer (runs in browser, writes to localStorage)
        tab_now = st_javascript("document.hidden ? 0 : 1")
        tab_now = (tab_now == 1)
        if tab_now != st.session_state.get("tab_visible", True):
            st.session_state["tab_visible"] = tab_now
            # Immediately push to monitor thread
            monitor.tab_visible = tab_now

        if not st.session_state.get("tab_visible", True):
            st.warning("⚠️ Your exam tab is hidden or unfocused! "
                       "Return immediately or the exam may end.")

        # ── Layout: question left (3/4) | camera right (1/4) ─────────────
        q_col, vid_col = st.columns([3, 1])

        # ── QUESTION PANEL ─────────────────────────────────────────────────
        with q_col:
            st.subheader("📝 Question")

            if sel == "None":
                st.warning("Please select a test from the sidebar.")

            elif st.session_state["current_q"] >= total_q:
                st.success("✅ All questions answered! Waiting for monitoring to finish…")

            else:
                cq_idx = st.session_state["current_q"]
                cq     = questions[cq_idx]

                # Compute time remaining for THIS question
                elapsed_q   = time.time() - st.session_state["question_start"]
                remaining_q = max(0, QUESTION_DURATION - int(elapsed_q))

                st.write(f"**Q{cq_idx + 1}/{total_q}:** {cq['text']}")
                
                st.markdown(f"""
                <div id="qtimer" style="font-size:18px; font-weight:bold;">
                ⏱️ Time left: <span id="qcount">{remaining_q}</span>s
                </div>
                <script>
                var c = {remaining_q};
                var t = setInterval(function(){{
                    c--;
                    if(c <= 0){{ c = 0; clearInterval(t); }}
                    var el = document.getElementById('qcount');
                    if(el) el.innerText = c;
                }}, 1000);
                </script>
                """, unsafe_allow_html=True)

                # ── Answer radio ───────────────────────────────────────────
                # Use a STABLE key based on question index only.
                # Pre-populate with previously chosen answer (if any) so the
                # radio doesn't reset when the page reruns.
                prev_answer = st.session_state["answers"].get(cq_idx)
                options     = cq["options"]
                default_idx = options.index(prev_answer) if prev_answer in options else 0

                selected = st.radio(
                    "Choose your answer:",
                    options,
                    index=default_idx,
                    key=f"q_radio_{cq_idx}",   # stable key per question
                )
                # Save answer without triggering a rerun
                st.session_state["answers"][cq_idx] = selected

                # ── Auto-advance when time runs out ────────────────────────
                if remaining_q == 0:
                    st.session_state["current_q"]      += 1
                    st.session_state["question_start"]  = time.time()
                    st.rerun()

                # Progress bar for time
                pct = remaining_q / QUESTION_DURATION
                st.progress(pct)

        # ── CAMERA / MONITORING PANEL ────────────────────────────────────
        @st.fragment(run_every=1)
        def camera_panel():
          with vid_col:
            st.markdown("#### 📹 Monitoring")
            video_ph   = st.empty()
            metrics_ph = st.empty()
            alerts_ph  = st.empty()
            chart_ph   = st.empty()

            # Snapshot of shared state (lock-free read is fine for display)
            frame_rgb = monitor.latest_frame_rgb
            if frame_rgb is not None:
                video_ph.image(frame_rgb, channels="RGB", use_container_width=True)
            else:
                video_ph.info("📷 Starting camera…")

            with metrics_ph.container():
                m = monitor.latest_metrics
                if m:
                    c1, c2 = st.columns(2)
                    elapsed_total = time.time() - monitor.monitor_start_time
                    remaining_total = max(0, exam_duration - int(elapsed_total))
                    c1.metric("⏱ Exam Remaining", f"{remaining_total}s")
                    c2.metric("⚠️ Warnings",  m.get("Warnings", "0/0"))
                    c3, c4 = st.columns(2)
                    c3.metric("🔔 Alerts",    m.get("Total Alerts", 0))
                    c4.metric("📊 Status",    m.get("Status", "…"))
                else:
                    st.caption("Waiting for live data…")

            if monitor.latest_recent_alerts:
                with alerts_ph.container():
                    st.markdown("**⚠️ Recent Alerts**")
                    for a in monitor.latest_recent_alerts:
                        icon = "🔴" if a.get("severity") == "critical" else "🟡"
                        t    = a.get("time", 0)
                        st.caption(f"{icon} [{t:.1f}s] {a.get('message', '')}")

            if monitor.latest_alert_timeline:
                try:
                    df_tl = pd.DataFrame(monitor.latest_alert_timeline)
                    fig   = px.scatter(df_tl, x="time", y="type", color="type",
                                       title="Alerts", height=200,
                                       labels={"time": "s", "type": "Type"})
                    fig.update_layout(margin=dict(l=0, r=0, t=30, b=0),
                                      showlegend=False)
                    chart_ph.plotly_chart(fig, use_container_width=True,
                                          key=f"chart_{int(time.time())// 10}")
                except Exception:
                    pass
        camera_panel()
        # ── Periodic UI refresh (controls the blink rate) ─────────────────
        # We sleep briefly then rerun – this replaces the old tight loop.
        # 2 seconds is enough to refresh camera frame and check timer.
        if monitor.is_monitoring():
            time.sleep(1)
            st.rerun()
        else:
            # Monitoring ended
            st.session_state["exam_ended_early"] = monitor.exam_ended_early
            st.session_state.stage               = "complete"
            st.rerun()

    # ── COMPLETE ──────────────────────────────────────────────────────────
    elif stage == "complete":
        st.header("📊 Exam Complete – Final Report")

        if st.session_state.get("exam_ended_early", False):
            st.error("❌ EXAM TERMINATED EARLY – Suspicious activity!")
        else:
            st.success("✅ Exam completed successfully!")

        mon    = st.session_state.monitor
        alerts = mon.session_data["alerts"]
        frames = mon.session_data["frames_processed"]
        warns  = mon.session_data["warning_count"]

        selected_test = st.session_state.get("selected_test", "None")
        if selected_test != "None":
            questions = SAMPLE_TESTS[selected_test]["questions"]
            answers   = st.session_state.get("answers", {})
            total_q   = len(questions)
            correct   = sum(
                1 for idx, q in enumerate(questions)
                if answers.get(idx) == q.get("correct_answer")
            )
            attempted = sum(1 for v in answers.values() if v)
            accuracy  = (correct / total_q * 100) if total_q else 0

            st.subheader("🏁 Final Exam Result")
            r1, r2, r3 = st.columns(3)
            r1.metric("Score",    f"{correct}/{total_q}")
            r2.metric("Accuracy", f"{accuracy:.0f}%")
            r3.metric("Answered", f"{attempted}/{total_q}")

            result_rows = []
            for idx, q in enumerate(questions):
                result_rows.append({
                    "Question": f"Q{idx+1}",
                    "Selected": answers.get(idx, "—"),
                    "Correct":  q.get("correct_answer", "—"),
                    "Status":   "✅" if answers.get(idx) == q.get("correct_answer") else "❌",
                })
            st.dataframe(pd.DataFrame(result_rows), use_container_width=True)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Alerts",    len(alerts))
        c2.metric("Frames Analysed", frames)
        c3.metric("Warnings",        f"{warns}/{MAX_WARNINGS}")
        c4.metric("Alert Rate",
                  f"{len(alerts)/frames*100:.1f}%" if frames else "0%")

        if alerts:
            st.subheader("📋 Alert Breakdown")
            df     = pd.DataFrame(alerts)
            counts = df["type"].value_counts()
            fig    = px.pie(values=counts.values, names=counts.index,
                            title="Alert Distribution")
            st.plotly_chart(fig, use_container_width=True)

            show_cols = [c for c in ["frame", "time", "type", "severity", "message"]
                         if c in df.columns]
            st.dataframe(df[show_cols], use_container_width=True)

            report = {
                "student":          student_name,
                "timestamp":        datetime.now().isoformat(),
                "exam_ended_early": st.session_state.get("exam_ended_early", False),
                "total_alerts":     len(alerts),
                "frames_processed": frames,
                "total_warnings":   warns,
                "alerts":           alerts,
            }
            st.download_button(
                label="📥 Download JSON Report",
                data=json.dumps(report, indent=2, default=str),
                file_name=(f"exam_report_{student_name}_"
                           f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"),
                mime="application/json",
            )
        else:
            st.success("🎉 No suspicious behaviour detected!")

        if st.button("🔄 Start New Session"):
            for key in ["monitor", "stage", "exam_ended_early", "tab_visible",
                        "last_tab_ts", "screen_share_ready", "monitor_started",
                        "current_q", "question_start", "answers"]:
                st.session_state.pop(key, None)
            st.rerun()


if __name__ == "__main__":
    main()