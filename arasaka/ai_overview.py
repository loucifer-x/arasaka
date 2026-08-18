import asyncio
import base64

import aiohttp
import cv2


# ============================================================
# OLLAMA CONFIG
# ============================================================

OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
OLLAMA_MODEL = "moondream"

OLLAMA_TIMEOUT = aiohttp.ClientTimeout(
    total=180,
    connect=10,
)


# ============================================================
# IMAGE ENCODING
# ============================================================

def _encode_face_to_b64(face_crop):
    """
    Convert an OpenCV BGR image into JPEG -> base64.
    """

    if face_crop is None:
        return None

    if face_crop.size == 0:
        return None

    try:
        success, buffer = cv2.imencode(
            ".jpg",
            face_crop,
            [
                cv2.IMWRITE_JPEG_QUALITY,
                90,
            ],
        )

        if not success:
            return None

        return base64.b64encode(
            buffer.tobytes()
        ).decode("utf-8")

    except Exception as e:
        print(f"Image encoding error: {e}")
        return None


# ============================================================
# ASYNC AI OVERVIEW
# ============================================================

async def generate_ai_overview(
    face_crop,
    session=None,
):
    """
    Generate an AI description of a face image using Ollama.

    This function is intentionally async so it doesn't freeze
    the DearPyGui interface while Ollama is processing.
    """

    if face_crop is None:
        return "No face image available."

    if face_crop.size == 0:
        return "No face image available."

    # --------------------------------------------------------
    # Encode image
    # --------------------------------------------------------

    image_b64 = await asyncio.to_thread(
        _encode_face_to_b64,
        face_crop,
    )

    if image_b64 is None:
        return "Could not encode face image."

    # --------------------------------------------------------
    # Prompt
    # --------------------------------------------------------

    prompt = """
You are analyzing a surveillance camera image.

Describe ONLY visible, non-sensitive information.

Mention things such as:
- clothing
- colours
- accessories
- visible objects
- pose
- general facial expression if clearly visible
- image quality
- lighting
- whether the face is partially obscured

Do NOT:
- identify the person
- guess their identity
- infer age
- infer ethnicity
- infer race
- infer religion
- infer sexuality
- infer medical conditions
- infer criminal behaviour
- make assumptions about things that cannot clearly be seen

Keep the response concise, around 2-4 sentences.

Describe what is actually visible in the image.
"""

    payload = {
        "model": OLLAMA_MODEL,

        "prompt": prompt,

        "images": [
            image_b64
        ],

        "stream": False,

        "options": {
            "temperature": 0.2,
        },
    }

    # --------------------------------------------------------
    # Create/reuse HTTP session
    # --------------------------------------------------------

    owns_session = False

    if session is None:
        session = aiohttp.ClientSession(
            timeout=OLLAMA_TIMEOUT
        )
        owns_session = True

    try:

        print(
            f"[AI] Sending image to Ollama "
            f"using {OLLAMA_MODEL}..."
        )

        async with session.post(
            OLLAMA_URL,
            json=payload,
        ) as response:

            # ------------------------------------------------
            # Read response body BEFORE raising an exception.
            # This is important because Ollama's useful error
            # message is otherwise hidden behind "500".
            # ------------------------------------------------

            response_text = await response.text()

            if response.status != 200:

                print(
                    "\n================ OLLAMA ERROR ================"
                )

                print(
                    f"HTTP status: {response.status}"
                )

                print(
                    f"Response:\n{response_text}"
                )

                print(
                    "==============================================\n"
                )

                return (
                    f"AI error: Ollama returned "
                    f"HTTP {response.status}: "
                    f"{response_text}"
                )

            # ------------------------------------------------
            # Parse JSON
            # ------------------------------------------------

            try:
                data = await response.json()

            except Exception as e:

                print(
                    f"[AI] Could not parse Ollama JSON: {e}"
                )

                print(
                    f"[AI] Raw response: {response_text}"
                )

                return "AI error: invalid Ollama response."

            # ------------------------------------------------
            # Extract response
            # ------------------------------------------------

            result = data.get(
                "response",
                ""
            )

            if not result:

                print(
                    "[AI] Ollama returned no response."
                )

                print(
                    f"[AI] Full response: {data}"
                )

                return "No AI overview was returned."

            print(
                "[AI] Overview generated successfully."
            )

            return result.strip()

    except asyncio.TimeoutError:

        print(
            "[AI] Ollama request timed out."
        )

        return (
            "AI error: Ollama timed out "
            "while generating the overview."
        )

    except aiohttp.ClientConnectionError as e:

        print(
            f"[AI] Could not connect to Ollama: {e}"
        )

        return (
            "AI error: Could not connect to Ollama. "
            "Make sure Ollama is running."
        )

    except Exception as e:

        print(
            f"[AI] Unexpected Ollama error: {type(e).__name__}: {e}"
        )

        return (
            f"AI error: {type(e).__name__}: {e}"
        )

    finally:

        if owns_session:

            await session.close()


# ============================================================
# SYNCHRONOUS HELPER
# ============================================================

def generate_ai_overview_sync(face_crop):
    """
    Synchronous wrapper.

    Useful because your DearPyGui callbacks are normal
    synchronous Python functions.

    IMPORTANT:
    Do NOT call this directly from the render loop if you
    want the UI to remain responsive.
    """

    return asyncio.run(
        generate_ai_overview(face_crop)
    )


# ============================================================
# BATCH PROCESSING
# ============================================================

async def generate_ai_overviews_batch(
    face_crops,
):
    """
    Generate AI descriptions for multiple images.

    One Ollama connection is shared.
    """

    if not face_crops:
        return []

    async with aiohttp.ClientSession(
        timeout=OLLAMA_TIMEOUT
    ) as session:

        tasks = [
            generate_ai_overview(
                crop,
                session=session,
            )
            for crop in face_crops
        ]

        return await asyncio.gather(
            *tasks
        )