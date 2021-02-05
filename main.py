import numpy as np
import face_recognition as fr
import cv2

video_capture = cv2.VideoCapture(0)

varun_image = fr.load_image_file("varun.jpg")
varun_face_encoding = fr.face_encodings(varun_image)[0]

anerudh_image = fr.load_image_file("anerudh.jpg")
anerudh_face_encoding = fr.face_encodings(anerudh_image)[0]

known_face_encondings = [varun_face_encoding, anerudh_face_encoding]
known_face_names = ["Varun", "Anerudh"]

while True: 
    ret, frame = video_capture.read()

    rgb_frame = frame[:, :, ::-1]

    face_locations = fr.face_locations(rgb_frame)
    face_encodings = fr.face_encodings(rgb_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):

        matches = fr.compare_faces(known_face_encondings, face_encoding)

        name = "Unknown"

        face_distances = fr.face_distance(known_face_encondings, face_encoding)

        best_match_index = np.argmin(face_distances)
        
        if matches[best_match_index]:
            name = known_face_names[best_match_index]
        
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 1)
        cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_TRIPLEX , 1.0, (255, 255, 255), 1)
        cv2.imshow('face_recog', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()