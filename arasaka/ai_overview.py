import asyncio
import cv2
import ollama


# ============================================================
# OLLAMA CONFIG
# ============================================================

OLLAMA_MODEL = "qwen3-vl:2b"


# ============================================================
# ASYNC AI OVERVIEW
# ============================================================

async def generate_ai_overview(
    face_crop,
):
    """
    Generate an AI description of a face image using Ollama.

    Ollama's Python library handles the connection and image
    encoding internally.
    """

    if face_crop is None:
        return "No face image available."

    if face_crop.size == 0:
        return "No face image available."

    # --------------------------------------------------------
    # Encode OpenCV image to JPEG bytes
    # --------------------------------------------------------

    try:

        success, buffer = await asyncio.to_thread(
            cv2.imencode,
            ".jpg",
            face_crop,
            [
                cv2.IMWRITE_JPEG_QUALITY,
                90,
            ],
        )

        if not success:
            return "Could not encode face image."

        image_bytes = buffer.tobytes()

    except Exception as e:

        print(f"[AI] Image encoding error: {e}")

        return f"Image encoding error: {e}"

    # --------------------------------------------------------
    # Prompt
    # --------------------------------------------------------

    prompt = """
Describe the face precisely using only visible features.

Include:
- face/head shape
- skin tone and texture
- eyes: shape, size, spacing, gaze
- eyebrows: shape and thickness
- nose: size and shape
- lips/mouth: shape and thickness
- cheeks, jawline, and chin
- hair and facial hair
- head direction
- race

Be objective and specific. Do not guess. If something is unclear, say "not visible".
"""

    # --------------------------------------------------------
    # Send image to Ollama
    # --------------------------------------------------------

    try:

        print(
            f"[AI] Sending image to Ollama "
            f"using {OLLAMA_MODEL}..."
        )

        response = await asyncio.to_thread(
            ollama.chat,
            model=OLLAMA_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                    "images": [
                        image_bytes
                    ],
                }
            ],
            options={
                "temperature": 0.2,
            },
        )

        # ----------------------------------------------------
        # Extract response
        # ----------------------------------------------------

        result = response["message"]["content"]

        if not result:

            print(
                "[AI] Ollama returned no response."
            )

            return "No AI overview was returned."

        print(
            "[AI] Overview generated successfully."
        )

        return result.strip()

    # --------------------------------------------------------
    # Error handling
    # --------------------------------------------------------

    except Exception as e:

        print(
            f"[AI] Ollama error: "
            f"{type(e).__name__}: {e}"
        )

        return (
            f"AI error: {type(e).__name__}: {e}"
        )


# ============================================================
# SYNCHRONOUS HELPER
# ============================================================

def generate_ai_overview_sync(face_crop):
    """
    Synchronous wrapper.

    Useful for normal Python callbacks.

    Do NOT call this inside a high-frequency render loop,
    because Ollama inference can take some time.
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
    """

    if not face_crops:
        return []

    tasks = [
        generate_ai_overview(
            crop
        )
        for crop in face_crops
    ]

    return await asyncio.gather(
        *tasks
    )
