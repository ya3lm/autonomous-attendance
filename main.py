import cv2
from deepface import DeepFace

# Open the webcam (0 = default camera)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    try:
        # Analyze the frame for age, gender, and race
        result = DeepFace.analyze(
            frame,
            actions=['age', 'gender', 'race'],
            enforce_detection=False,
            detector_backend='retinaface',
            model_name='Facenet'
        )

        # Extract the results
        age = result[0]['age']
        gender = result[0]['gender']
        dominant_race = result[0]['dominant_race']

        # Display the results on the frame
        text = f"Age: {age}  Gender: {gender}  Race: {dominant_race}"
        cv2.putText(frame, text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    except Exception as e:
        # If no face detected, ignore
        pass

    # Show the frame
    cv2.imshow("DeepFace Webcam", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()