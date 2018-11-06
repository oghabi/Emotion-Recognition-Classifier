import cv2
import sys
import dlib
import numpy as np

def real_time_face_tracker():
    predictor_path = 'shape_predictor_68_face_landmarks.dat'

    face_detector = dlib.get_frontal_face_detector()
    points_fitter = dlib.shape_predictor(predictor_path)

    video_capture = cv2.VideoCapture(0)

    itr, face_points = 1, []
    while True:
        ret, frame = video_capture.read()

        if not itr %2:
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            face_boxes, face_points = fit_face_points(gray, face_detector, points_fitter)

        if len(face_points):
            for (x, y) in face_points:
                cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)

        flipped = cv2.flip(frame, 1)
        cv2.imshow('Video', flipped)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        itr += 1

    video_capture.release()
    cv2.destroyAllWindows()

def shape_to_np(shape):
    coords = np.zeros((shape.num_parts, 2), dtype=int)
    for i in range(0, shape.num_parts):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords


def fit_face_points(img, face_detector, points_fitter):
    face_boxes, face_points = face_detector(img, 1), []

    if len(face_boxes):
        face_box = face_boxes[0]
        face_points = shape_to_np(points_fitter(img, face_box))

    return face_boxes, face_points

if __name__ == '__main__':
    real_time_face_tracker()
