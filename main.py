import cv2
from deepface import DeepFace
import os

KNOWN_FACES_DIR = "faces_database/students"

known_faces = []
for file in os.listdir(KNOWN_FACES_DIR):
    if file.lower().endswith(('.jpg', '.png', '.jpeg')):
        name = os.path.splitext(file)[0]
        path = os.path.join(KNOWN_FACES_DIR, file)
        known_faces.append((name, path))

if not known_faces:
    print("No faces found in database folder!")
    exit()

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

print("Loaded known faces:", [name for name, _ in known_faces])
print("Press 'q' to quit...")

# --- NEW VARIABLES FOR FRAME SKIPPING ---
frame_count = 0
process_frame_frequency = 10  # Process face recognition every 10th frame
recognized_name = None          # Stores the last recognized name
# --- END NEW VARIABLES ---

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 1. Increment the frame count
    frame_count += 1
    
    # 2. Check if it's time to run the expensive face recognition
    if frame_count % process_frame_frequency == 0:
        
        # Reset current name to 'Unknown' before trying to detect
        current_recognized_name = None
        
        try:

            for name, path in known_faces:
                result = DeepFace.verify(
                    img1_path=frame,
                    img2_path=path,
                    model_name="VGG-Face",
                    # *** Use a faster backend like 'opencv' or 'ssd' here ***
                    detector_backend="opencv", 
                    enforce_detection=False
                )
                
                print(f"Comparing with {name} (Frame {frame_count}): Verified={result['verified']}, Distance={result['distance']:.4f}")

                if result["verified"]:
                    current_recognized_name = name
                    break
            
            # 3. Update the global variable with the new result
            recognized_name = current_recognized_name

            # 4. Delete the temporary frame to save storage and improve performance
            #try:
            #    os.remove("current_frame.jpg")
            #except Exception as e:
            #    print(f"Warning: Could not delete temporary frame - {e}")

        except Exception as e:
            # This catch is mainly for DeepFace errors, keep the last recognized_name
            print(f"DeepFace Error (Frame {frame_count}): {e}")
    
    # --- DISPLAY LOGIC (RUNS ON EVERY FRAME) ---
    # Use the last successful result for every frame
    if recognized_name:
        text = f"Recognized: {recognized_name}"
        color = (0, 255, 0)
    else:
        text = "Unknown face"
        color = (0, 0, 255)

    cv2.putText(frame, text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    cv2.imshow("Autonomous Attendance", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()