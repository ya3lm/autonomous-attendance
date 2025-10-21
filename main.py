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

while True:
    ret, frame = cap.read()
    if not ret:
        break

    try:
        cv2.imwrite("current_frame.jpg", frame)
        recognized_name = None

        for name, path in known_faces:
            result = DeepFace.verify(
                img1_path="current_frame.jpg",
                img2_path=path,
                model_name="VGG-Face",
                detector_backend="yolov12s",
                enforce_detection=False
            )
            print(f"Comparing with {name}: Verified={result['verified']}, Distance={result['distance']:.4f}")

            if result["verified"]:
                recognized_name = name
                break

        if recognized_name:
            text = f"Recognized: {recognized_name}"
            color = (0, 255, 0)
        else:
            text = "Unknown face"
            color = (0, 0, 255)

        cv2.putText(frame, text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    except Exception as e:
        print("Error:", e)

    cv2.imshow("Autonomous Attendance", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()