import cv2

print("Opening camera... Press 'q' to quit")

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("❌ Error: Could not open camera")
    exit()

while True:
    ret, frame = cap.read()
    
    if not ret:
        print("❌ Error: Can't receive frame")
        break
    
    cv2.imshow('Camera Test - Press Q to quit', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("✓ Camera test completed!")