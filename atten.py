import cv2
import face_recognition
import numpy as np
import os
import time

# Set up attendance log file
log_file = 'attendance_log.txt'

# Load known faces and names
known_faces_dir = 'known_faces'
known_faces_encodings = []
known_faces_names = []
for filename in os.listdir(known_faces_dir):
    image = face_recognition.load_image_file(os.path.join(known_faces_dir, filename))
    encoding = face_recognition.face_encodings(image)[0]
    known_faces_encodings.append(encoding)
    known_faces_names.append(os.path.splitext(filename)[0])

# Initialize webcam
video_capture = cv2.VideoCapture(0)

# Start loop for detecting and recognizing faces
while True:
    # Capture frame from webcam
    ret, frame = video_capture.read()
    
    # Resize frame to 1/4 size for faster processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    
    # Convert BGR color to RGB color
    rgb_small_frame = small_frame[:, :, ::-1]
    
    # Find all the faces and face encodings in the current frame
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
    
    # Loop through each face in the current frame
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # Check if the face is a match for the known faces
        matches = face_recognition.compare_faces(known_faces_encodings, face_encoding)
        
        # Find the best match index
        face_distances = face_recognition.face_distance(known_faces_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        
        # If the face is a match, log the attendance
        if matches[best_match_index]:
            name = known_faces_names[best_match_index]
            with open(log_file, 'a') as f:
                f.write(f'{name} - {time.strftime("%Y-%m-%d %H:%M:%S")}\n')
                
        # Draw a box around the face and label it with the name
        cv2.rectangle(frame, (left*4, top*4), (right*4, bottom*4), (0, 255, 0), 2)
        cv2.rectangle(frame, (left*4, bottom*4 - 35), (right*4, bottom*4), (0, 255, 0), cv2.FILLED)
        cv2.putText(frame, name, (left*4 + 6, bottom*4 - 6), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # Display the resulting image
    cv2.imshow('Video', frame)
    
    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release webcam and close window
video_capture.release()
cv2.destroyAllWindows()