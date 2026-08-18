import cv2
import face_recognition


def get_face_data(face_crop, face_location):
    if face_crop is None or face_crop.size == 0:
        return None

    face_rgb = cv2.cvtColor(
        face_crop,
        cv2.COLOR_BGR2RGB
    )

    top, right, bottom, left = face_location

    # Make sure coordinates are valid
    h, w = face_rgb.shape[:2]

    top = max(0, min(top, h - 1))
    bottom = max(top + 1, min(bottom, h))
    left = max(0, min(left, w - 1))
    right = max(left + 1, min(right, w))

    face_location = (
        top,
        right,
        bottom,
        left,
    )

    encodings = face_recognition.face_encodings(
        face_rgb,
        known_face_locations=[face_location],
        num_jitters=1,
        model="small",
    )

    if not encodings:
        return None

    return encodings[0]