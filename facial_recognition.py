import cv2
from deepface import DeepFace

def analyze_webcam():
    # Start webcam capture (default is 0 for the primary webcam)
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Unable to access the camera.")
        return
    
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        if not ret:
            print("Failed to grab frame.")
            break
        
        # Analyze the frame for gender and age
        try:
            result = DeepFace.analyze(frame, actions=['gender', 'age'])
            
            # Extract the dominant gender and age from the result
            gender = result[0]['dominant_gender']
            age = result[0]['age']
            
            # Display the results at the top of the frame
            cv2.putText(frame, f"Gender: {gender}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Age: {age}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        except Exception as e:
            print(f"Error analyzing frame: {e}")
        
        # Show the current frame with the text
        cv2.imshow('Webcam Feed', frame)
        
        # Break the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release the webcam and close the OpenCV window
    cap.release()
    cv2.destroyAllWindows()

# Run the webcam face recognition
analyze_webcam()
