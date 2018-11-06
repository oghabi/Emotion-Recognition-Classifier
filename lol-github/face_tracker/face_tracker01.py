import cv2
import sys

face_casc_path = "haarcascade_frontalface_default.xml"
smile_casc_path = "haarcascade_smile.xml"
predictor_path = "shape_predictor_68_face_landmarks.dat"

face_cascade = cv2.CascadeClassifier(face_casc_path)
smile_cascade = cv2.CascadeClassifier(smile_casc_path)

video_capture = cv2.VideoCapture(0)

while True:
    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #landmarks = stasm.search_single(img)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(50, 50),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    for (x, y, w, h) in faces:
        face_cropped = frame[y:y+h, x:x+w, :]

        smiles = smile_cascade.detectMultiScale(
            face_cropped,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(70, 70),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        for (x2, y2, w2, h2) in smiles:
            cv2.rectangle(frame, (x+x2, y+y2), (x+x2+w2, y+y2+h2), (255, 0, 0), 2)

    flipped = cv2.flip(frame, 1)
    cv2.imshow('Video', flipped)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
