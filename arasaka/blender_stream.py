import cv2
import numpy as np
import requests


class BlenderStream:

    def __init__(
        self,
        url,
        boundary=None,
        connect_timeout=10,
        read_timeout=5
    ):
        self.url = url
        self.response = None
        self.closed = False

        try:
            self.response = requests.get(
                url,
                stream=True,
                timeout=(connect_timeout, read_timeout),
                headers={
                    "User-Agent": "Blender-Live-View-Client",
                    "Accept": (
                        "multipart/x-mixed-replace,"
                        "image/jpeg,"
                        "image/*,"
                        "*/*"
                    ),
                },
            )

            self.response.raise_for_status()

        except requests.RequestException as e:
            raise RuntimeError(
                f"Could not connect to camera stream '{url}': {e}"
            ) from e

        content_type = self.response.headers.get(
            "Content-Type",
            ""
        ).lower()

        print("Connected to stream")
        print("URL:", url)
        print("Content-Type:", content_type)

        # --------------------------------------------------------
        # Determine stream type
        # --------------------------------------------------------

        if content_type.startswith(
            "multipart/x-mixed-replace"
        ):
            self.stream_type = "multipart"

            self.boundary = (
                boundary
                or self._parse_boundary(content_type)
            )

            if self.boundary is None:
                self.boundary = "frame"

                print(
                    "WARNING: Multipart stream has no boundary. "
                    "Using 'frame'."
                )

            # Handle boundary values such as:
            #
            # boundary=frame
            # boundary="frame"
            # boundary=--frame
            #
            self.boundary = self.boundary.strip().strip('"')

            if self.boundary.startswith("--"):
                self.boundary = self.boundary[2:]

            self.boundary_bytes = (
                b"--" + self.boundary.encode()
            )

            print(
                "Using boundary:",
                self.boundary_bytes
            )

        elif (
            content_type.startswith("image/jpeg")
            or content_type.startswith("image/jpg")
        ):
            # Single JPEG response.
            self.stream_type = "jpeg"

            print(
                "Detected single JPEG response."
            )

        elif "text/html" in content_type:
            # This is NOT a video stream.
            self.close()

            raise RuntimeError(
                f"Camera returned HTML instead of a video stream: "
                f"{url}\n"
                f"Content-Type: {content_type}\n"
                f"The URL is probably the camera's web page, "
                f"not its video endpoint."
            )

        else:
            # Some servers don't provide a useful Content-Type.
            #
            # We'll attempt to detect JPEG frames from the raw stream.
            self.stream_type = "raw"

            print(
                "WARNING: Unknown Content-Type. "
                "Attempting raw JPEG frame detection."
            )

    # ============================================================
    # Boundary parser
    # ============================================================

    @staticmethod
    def _parse_boundary(content_type):
        """
        Extract boundary from:

            multipart/x-mixed-replace; boundary=frame

        or:

            multipart/x-mixed-replace; boundary="frame"
        """

        for part in content_type.split(";"):

            part = part.strip()

            if part.lower().startswith("boundary="):

                boundary = part.split(
                    "=",
                    1
                )[1].strip()

                boundary = boundary.strip('"')

                if boundary.startswith("--"):
                    boundary = boundary[2:]

                return boundary

        return None

    # ============================================================
    # Frame generator
    # ============================================================

    def frames(self):

        if self.response is None:
            return

        if self.stream_type == "multipart":
            yield from self._multipart_frames()

        elif self.stream_type == "jpeg":
            frame = self._read_single_jpeg()

            if frame is not None:
                yield frame

        elif self.stream_type == "raw":
            yield from self._raw_jpeg_frames()

    # ============================================================
    # Multipart MJPEG
    # ============================================================

    def _multipart_frames(self):

        buffer = b""

        try:

            for chunk in self.response.iter_content(
                chunk_size=65536
            ):

                if self.closed:
                    break

                if not chunk:
                    continue

                buffer += chunk

                while True:

                    # --------------------------------------------
                    # Find boundary
                    # --------------------------------------------

                    boundary_pos = buffer.find(
                        self.boundary_bytes
                    )

                    if boundary_pos == -1:

                        # Prevent unlimited memory growth if the
                        # server sends unexpected data.
                        if len(buffer) > 2_000_000:
                            buffer = buffer[-100_000:]

                        break

                    # --------------------------------------------
                    # Find HTTP headers
                    # --------------------------------------------

                    header_end = buffer.find(
                        b"\r\n\r\n",
                        boundary_pos
                    )

                    if header_end == -1:
                        break

                    headers = buffer[
                        boundary_pos:header_end
                    ]

                    image_start = header_end + 4

                    # --------------------------------------------
                    # Content-Length
                    # --------------------------------------------

                    content_length = (
                        self._get_content_length(headers)
                    )

                    if content_length is not None:

                        image_end = (
                            image_start
                            + content_length
                        )

                        if len(buffer) < image_end:
                            break

                        image_data = buffer[
                            image_start:image_end
                        ]

                        buffer = buffer[
                            image_end:
                        ]

                    else:

                        # ----------------------------------------
                        # No Content-Length
                        # Find next boundary.
                        # ----------------------------------------

                        next_boundary = buffer.find(
                            self.boundary_bytes,
                            image_start
                        )

                        if next_boundary == -1:
                            break

                        image_data = buffer[
                            image_start:next_boundary
                        ].rstrip(b"\r\n")

                        buffer = buffer[
                            next_boundary:
                        ]

                    # --------------------------------------------
                    # Decode JPEG
                    # --------------------------------------------

                    frame = self._decode(
                        image_data
                    )

                    if frame is not None:
                        yield frame

        except (
            requests.exceptions.ReadTimeout,
            requests.exceptions.ConnectionError,
            requests.exceptions.ChunkedEncodingError
        ) as e:

            if not self.closed:
                print(
                    f"Stream interrupted for {self.url}: {e}"
                )

    # ============================================================
    # Single JPEG
    # ============================================================

    def _read_single_jpeg(self):

        try:

            data = self.response.content

            return self._decode(data)

        except Exception as e:

            print(
                f"Could not decode JPEG from {self.url}: {e}"
            )

            return None

    # ============================================================
    # Raw JPEG stream
    #
    # Finds:
    #
    # FF D8 = JPEG START
    # FF D9 = JPEG END
    # ============================================================

    def _raw_jpeg_frames(self):

        buffer = b""

        try:

            for chunk in self.response.iter_content(
                chunk_size=65536
            ):

                if self.closed:
                    break

                if not chunk:
                    continue

                buffer += chunk

                while True:

                    start = buffer.find(
                        b"\xff\xd8"
                    )

                    if start == -1:

                        # Prevent unlimited growth.
                        if len(buffer) > 2_000_000:
                            buffer = buffer[-100_000:]

                        break

                    end = buffer.find(
                        b"\xff\xd9",
                        start + 2
                    )

                    if end == -1:
                        break

                    end += 2

                    image_data = buffer[
                        start:end
                    ]

                    buffer = buffer[end:]

                    frame = self._decode(
                        image_data
                    )

                    if frame is not None:
                        yield frame

        except (
            requests.exceptions.ReadTimeout,
            requests.exceptions.ConnectionError,
            requests.exceptions.ChunkedEncodingError
        ) as e:

            if not self.closed:
                print(
                    f"Raw stream interrupted for "
                    f"{self.url}: {e}"
                )

    # ============================================================
    # Content-Length
    # ============================================================

    @staticmethod
    def _get_content_length(headers):

        for line in headers.split(b"\r\n"):

            if line.lower().startswith(
                b"content-length:"
            ):

                try:

                    return int(
                        line.split(
                            b":",
                            1
                        )[1].strip()
                    )

                except ValueError:
                    return None

        return None

    # ============================================================
    # JPEG decoder
    # ============================================================

    @staticmethod
    def _decode(image_data):

        if not image_data:
            return None

        try:

            array = np.frombuffer(
                image_data,
                dtype=np.uint8
            )

            frame = cv2.imdecode(
                array,
                cv2.IMREAD_COLOR
            )

            return frame

        except Exception as e:

            print(
                f"Frame decode error: {e}"
            )

            return None

    # ============================================================
    # Close
    # ============================================================

    def close(self):

        self.closed = True

        if self.response is not None:

            try:
                self.response.close()

            except Exception:
                pass

            self.response = None
