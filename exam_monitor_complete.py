def monitor_exam(self, duration, student_name, video_placeholder, metrics_placeholder, 
                     alerts_placeholder, chart_placeholder):
        """Monitor exam with live updates"""
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        
        self.session_data['start_time'] = datetime.now()
        start_time = time.time()
        frame_number = 0
        alert_cooldown = {}
        alert_timeline = []
        
        while time.time() - start_time < duration:
            ret, frame = cap.read()
            if not ret:
                continue
            
            frame_number += 1
            elapsed = time.time() - start_time
            remaining = int(duration - elapsed)
            
            current_alerts = []
            
            try:
                result = DeepFace.analyze(frame, actions=['emotion'], 
                                         enforce_detection=False, silent=True)
                
                if isinstance(result, list):
                    result = result[0]
                
                face_region = result['region']
                x, y, w, h = face_region['x'], face_region['y'], face_region['w'], face_region['h']
                center_x, center_y = x + w // 2, y + h // 2
                
                emotions = result['emotion']
                dominant_emotion = result['dominant_emotion']
                
                # Check for deviations
                alerts = self.check_suspicious_behavior([center_x, center_y], emotions, frame_number)
                
                for alert in alerts:
                    alert_key = alert['type']
                    last_alert_time = alert_cooldown.get(alert_key, 0)
                    
                    if time.time() - last_alert_time > 5:
                        current_alerts.append(alert)
                        alert_cooldown[alert_key] = time.time()
                        self.session_data['alerts'].append(alert)
                        alert_timeline.append({
                            'time': elapsed,
                            'type': alert['type'],
                            'message': alert['message']
                        })
                
                # Draw on frame
                box_color = (0, 0, 255) if current_alerts else (0, 255, 0)
                cv2.rectangle(frame, (x, y), (x+w, y+h), box_color, 2)
                
                status_text = "⚠ ALERT" if current_alerts else "✓ Normal"
                status_color = (0, 0, 255) if current_alerts else (0, 255, 0)
                
                cv2.putText(frame, status_text, (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
                cv2.putText(frame, f"Time: {remaining}s", (10, 70),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, f"Emotion: {dominant_emotion}", (10, 100),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
            except Exception as e:
                cv2.putText(frame, "⚠ NO FACE DETECTED", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                # Continue monitoring even if face detection fails
                pass
            
            # Update display
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            video_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
            
            # Update metrics
            with metrics_placeholder.container():
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Time Remaining", f"{remaining}s")
                col2.metric("Total Alerts", len(self.session_data['alerts']))
                col3.metric("Frames Processed", frame_number)
                col4.metric("Current Status", "🔴 Alert" if current_alerts else "🟢 Normal")
            
            # Update alerts list
            if self.session_data['alerts']:
                with alerts_placeholder.container():
                    st.subheader("⚠️ Recent Alerts")
                    for alert in self.session_data['alerts'][-5:]:
                        st.warning(f"**Frame {alert['frame']}**: {alert['message']}")
            
            # Update chart - FIXED VERSION
            if alert_timeline and len(alert_timeline) > 0:
                try:
                    df = pd.DataFrame(alert_timeline)
                    fig = px.scatter(df, x='time', y='type', 
                                   title='Alert Timeline',
                                   labels={'time': 'Time (seconds)', 'type': 'Alert Type'},
                                   color='type')
                    # Added unique key to prevent duplicate element error
                    chart_placeholder.plotly_chart(fig, use_container_width=True, key=f"timeline_chart_{frame_number}")
                except Exception as e:
                    # If chart fails, just skip it - monitoring continues
                    pass
            
            self.session_data['frames_processed'] = frame_number
        
        cap.release()
        return alert_timeline