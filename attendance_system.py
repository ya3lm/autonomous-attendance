import cv2
from deepface import DeepFace
import os
from typing import List, Tuple, Optional

# --- Configuration ---
KNOWN_FACES_DIR = "faces_database/students"
PROCESS_FRAME_FREQUENCY = 10 # Process face recognition every 10th frame

# --- Core Functions ---
def load_known_faces(base_dir: str) -> List[Tuple[str, str]]:
    """
    Loads face images from the database directory.
    Returns a list of tuples: [(student_name, image_path), ...]
    """
    known_faces = []
    if not os.path.exists(base_dir):
        #print(f"Warning: Database directory not found at {base_dir}")
        return known_faces
        
    for student_name in os.listdir(base_dir):
        student_folder = os.path.join(base_dir, student_name)
        if os.path.isdir(student_folder):
            for file in os.listdir(student_folder):
                if file.lower().endswith(('.jpg', '.png', '.jpeg')):
                    image_path = os.path.join(student_folder, file)
                    known_faces.append((student_name, image_path))
    return known_faces

def identify_face_in_frame(frame, known_faces: List[Tuple[str, str]]) -> Optional[str]:
    """
    Iterates through known faces and attempts to verify the frame against them.
    Returns the name of the recognized student or None.
    """
    # The logic for verification goes here, separated from the camera loop.
    for name, path in known_faces:
        try:
            result = DeepFace.verify(
                img1_path=frame,
                img2_path=path,
                model_name="VGG-Face",
                detector_backend="opencv", 
                enforce_detection=False
            )
            
            # Print is for debugging purposes
            # print(f"Comparing with {name}: Verified={result['verified']}")
            
            if result["verified"]:
                return name  # Found a match, return the name immediately
        except Exception as e:
            # Handle DeepFace internal errors (e.g., no face detected in frame)
            # print(f"DeepFace Error: {e}")
            pass
            
    return None # No match found after checking all known faces

# --- Main Application Loop ---

def run_attendance_system():
    # 1. Initialization and Setup
    known_faces = load_known_faces(KNOWN_FACES_DIR)

    if not known_faces:
        print("No faces found in database folder! Exiting.")
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera. Exiting.")
        return
        
    # Initialization of loop variables
    frame_count = 0
    recognized_name: Optional[str] = None

    print(f"Loaded {len(known_faces)} known faces.")
    print("Press 'q' to quit...")
    
    # 2. Main Loop
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        
        # Process face recognition only every N frames
        if frame_count % PROCESS_FRAME_FREQUENCY == 0:
            newly_recognized_name = identify_face_in_frame(frame, known_faces)
            recognized_name = newly_recognized_name # Update the display name

        # 3. Display Logic
        if recognized_name:
            text = f"Recognized: {recognized_name}"
            color = (0, 255, 0)
        else:
            text = "Unknown face"
            color = (0, 0, 255)

        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.imshow("Autonomous Attendance", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 4. Cleanup
    cap.release()
    cv2.destroyAllWindows()

# This is the entry point, separate from the functions above
if __name__ == '__main__':
    run_attendance_system()