import cv2
import pywhatkit as kit
import time
import random

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)

print("Press 'q' to detect mood and play Telugu songs...")
print("Press ESC to exit")

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # ✅ Default values (fix error)
    brightness = gray.mean()
    contrast = gray.std()

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)

        face_roi = gray[y:y+h, x:x+w]
        brightness = face_roi.mean()
        contrast = face_roi.std()

    cv2.putText(frame, "Press Q to detect mood", (20,30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

    cv2.imshow("Smart Telugu Music System", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        print("\nDetecting mood...")

        # 🎯 Safe logic
        if contrast > 60:
            emotion = "happy"
        elif brightness < 70:
            emotion = "sad"
        else:
            emotion = random.choice(["happy", "sad", "calm"])

        print("Detected Mood:", emotion)

        # 🎵 NO ROMANTIC SONGS
        if emotion == "happy":
            options = [
                "telugu mass songs",
                "telugu dance songs",
                "telugu party songs"
            ]
        elif emotion == "sad":
            options = [
                "sad telugu songs",
                "telugu emotional songs",
                "telugu slow sad songs"
            ]
        else:
            options = [
                "telugu instrumental music",
                "relaxing telugu music",
                "telugu background music"
            ]

        search = random.choice(options)

        print("Playing:", search)

        kit.playonyt(search)
        time.sleep(5)

    elif key == 27:
        break

cap.release()
cv2.destroyAllWindows()