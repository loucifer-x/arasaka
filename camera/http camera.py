import argparse
import time

import cv2
from flask import Flask, Response

app = Flask(__name__)

CAMERA_INDEX = 0
JPEG_QUALITY = 80
BOUNDARY = "frame"


def generate_frames(camera_index=CAMERA_INDEX, jpeg_quality=JPEG_QUALITY):

    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        print(
            f"ERROR: Could not open camera index {camera_index}"
        )
        return

    print(
        f"Camera {camera_index} opened successfully"
    )

    boundary_bytes = f"--{BOUNDARY}".encode()

    try:

        while True:

            success, frame = cap.read()

            if not success:
                print(
                    "WARNING: frame grab failed, retrying..."
                )

                time.sleep(0.1)
                continue

            ok, buffer = cv2.imencode(
                ".jpg",
                frame,
                [
                    cv2.IMWRITE_JPEG_QUALITY,
                    jpeg_quality
                ],
            )

            if not ok:
                continue

            frame_bytes = buffer.tobytes()

            yield (
                boundary_bytes
                + b"\r\n"
                + b"Content-Type: image/jpeg\r\n"
                + b"Content-Length: "
                + str(len(frame_bytes)).encode()
                + b"\r\n"
                + b"\r\n"
                + frame_bytes
                + b"\r\n"
            )

    finally:

        cap.release()

        print(
            f"Camera {camera_index} released"
        )


@app.route("/")
def index():

    return """
    <html>
        <head>
            <title>Webcam Stream</title>
        </head>

        <body style="
            margin:0;
            background:#111;
            display:flex;
            align-items:center;
            justify-content:center;
            height:100vh;
        ">

            <img
                src="/video_feed"
                style="max-width:100%;max-height:100%;"
            >

        </body>
    </html>
    """


@app.route("/video_feed")
def video_feed():

    camera_index = app.config["CAMERA_INDEX"]
    jpeg_quality = app.config["JPEG_QUALITY"]

    # --------------------------------------------------------
    # Test camera BEFORE returning the streaming response.
    # --------------------------------------------------------

    test_cap = cv2.VideoCapture(camera_index)

    if not test_cap.isOpened():

        print(
            f"ERROR: Camera {camera_index} "
            f"could not be opened."
        )

        test_cap.release()

        return (
            f"Camera {camera_index} could not be opened",
            500
        )

    test_cap.release()

    print(
        f"Starting video feed for camera "
        f"{camera_index}"
    )

    return Response(
        generate_frames(
            camera_index,
            jpeg_quality
        ),
        mimetype=(
            "multipart/x-mixed-replace; "
            f"boundary={BOUNDARY}"
        ),
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Webcam MJPEG livestream server"
    )

    parser.add_argument(
        "--camera",
        type=int,
        default=CAMERA_INDEX
    )

    parser.add_argument(
        "--quality",
        type=int,
        default=JPEG_QUALITY
    )

    parser.add_argument(
        "--port",
        type=int,
        default=5000
    )

    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0"
    )

    args = parser.parse_args()

    app.config["CAMERA_INDEX"] = args.camera
    app.config["JPEG_QUALITY"] = args.quality

    print(
        "Starting camera server:"
    )

    print(
        f"  Camera: {args.camera}"
    )

    print(
        f"  Port:   {args.port}"
    )

    print(
        f"  Host:   {args.host}"
    )

    app.run(
        host=args.host,
        port=args.port,
        threaded=True
    )
