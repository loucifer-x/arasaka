import cv2

# ============================================================
# Face Detector (YuNet)
# ============================================================

MODEL_PATH = "face_detection_yunet.onnx"

face_detector = cv2.FaceDetectorYN.create(
    MODEL_PATH,
    "",
    (320, 320),          # input size, gets overridden by setInputSize below
    score_threshold=0.7,  # raise -> fewer false positives, lower -> more recall
    nms_threshold=0.3,
    top_k=5000
)


def detect_faces(frame):
    if frame is None:
        raise ValueError("detect_faces received an empty frame")

    h, w = frame.shape[:2]
    face_detector.setInputSize((w, h))

    _, faces = face_detector.detect(frame)

    if faces is None:
        return []

    results = []

    for f in faces:
        x, y, box_w, box_h = f[:4].astype(int)
        confidence = float(f[-1])

        # Original YuNet face rectangle
        face_x1 = max(0, x)
        face_y1 = max(0, y)
        face_x2 = min(w, x + box_w)
        face_y2 = min(h, y + box_h)

        # Padding around the face
        pad_w = int(box_w * 0.15)
        pad_top = int(box_h * 0.20)
        pad_bottom = int(box_h * 0.10)

        crop_x1 = max(0, x - pad_w)
        crop_y1 = max(0, y - pad_top)
        crop_x2 = min(w, x + box_w + pad_w)
        crop_y2 = min(h, y + box_h + pad_bottom)

        # Crop from original image
        crop = frame[
            crop_y1:crop_y2,
            crop_x1:crop_x2
        ]

        if crop.size == 0:
            continue

        # YuNet face box expressed relative to the crop
        top = face_y1 - crop_y1
        left = face_x1 - crop_x1
        bottom = face_y2 - crop_y1
        right = face_x2 - crop_x1

        results.append({
            "box": (
                crop_x1,
                crop_y1,
                crop_x2 - crop_x1,
                crop_y2 - crop_y1,
            ),

            "crop": crop,

            "face_location": (
                top,
                right,
                bottom,
                left,
            ),

            "confidence": confidence,
        })

    return results
