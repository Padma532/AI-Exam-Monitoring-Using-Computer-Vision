from deepface import DeepFace
import cv2

print("Testing DeepFace emotion detection... Press 'q' to quit")

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    try:
        result = DeepFace.analyze(frame, actions=['emotion'], 
                                 enforce_detection=False, silent=True)
        
        if isinstance(result, list):
            result = result[0]
        
        emotion = result['dominant_emotion']
        
        cv2.putText(frame, f"Emotion: {emotion}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        print(f"Detected: {emotion}")
        
    except Exception as e:
        cv2.putText(frame, "No face detected", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    cv2.imshow('DeepFace Test - Press Q to quit', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("✓ DeepFace test completed!")