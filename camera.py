import cv2
import time
import os

def open_camera():
    # Try different camera indices
    for camera_index in [0, 1]:
        model = cv2.CascadeClassifier(os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml'))
        cap = cv2.VideoCapture(camera_index)
        
        if cap.isOpened():
            # Test if we can actually read a frame
            ret, frame = cap.read()
            if ret:
                print(f"Successfully connected to camera {camera_index}")
                
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        print(f"Error: Lost connection to camera {camera_index}")
                        break
                    faces = model.detectMultiScale(frame)
                    for x, y, w, h in faces:
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        
                    # Display the frame
                    cv2.imshow('Camera Feed', frame)
                    
                    # Break the loop when 'q' is pressed
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
                cap.release()
                cv2.destroyAllWindows()
                return
            
        cap.release()
    
    print("Error: Could not connect to any camera")

if __name__ == "__main__":
    # Add a small delay to ensure camera initialization
    time.sleep(1)
    open_camera()
