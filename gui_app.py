import json
import queue
import threading
import time
from ai_overview import generate_ai_overview
import asyncio
import cv2
import face_recognition
import numpy as np
from PIL import Image, ImageTk
from face_detector import detect_faces

from config import (
    STREAM_URL,
    WINDOW_TITLE,
    VIDEO_WIDTH,
    FACE_THUMB_SIZE,
    GRID_COLUMNS,
    MAX_FACE_SLOTS,
    LIVE_SECTION_HEIGHT,
    SAVE_INTERVAL_SECONDS,
    HITS_MAX_ROWS,
)
from blender_stream import BlenderStream


import dearpygui.dearpygui as dpg

history_faces = []


# ============================================================
# CAMERA CONFIGURATION
# ============================================================

# ============================================================
# CAMERA CONFIGURATION
# ============================================================

from pathlib import Path
import json

# cameras.json is ALWAYS stored outside the Python code:
#
# main/
# ├── gui_app.py
# └── json/
#     └── cameras.json
#
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
CAMERAS_PATH = BASE_DIR / "cameras.json"


def load_cameras(path=None):
    """Load all camera information from the external JSON file."""

    if path is None:
        path = CAMERAS_PATH

    path = Path(path)

    empty_config = {
        "camera": {}
    }

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

    except FileNotFoundError:
        # Create the external json directory/file if it doesn't exist.
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(empty_config, f, indent=2)

        return empty_config

    except json.JSONDecodeError as e:
        print(f"Camera JSON is invalid: {e}")
        return empty_config

    except OSError as e:
        print(f"Could not read camera JSON: {e}")
        return empty_config

    # Make sure the JSON has the expected structure.
    if not isinstance(data, dict):
        print(f"Invalid camera configuration in '{path}'.")
        return empty_config

    if not isinstance(data.get("camera"), dict):
        print(f"Invalid camera configuration in '{path}'.")
        return empty_config

    return data



def save_cameras(data=None, path=None):
    """Save the camera configuration to the external JSON file."""

    if path is None:
        path = CAMERAS_PATH

    path = Path(path)

    if data is None:
        data = cameras

    # Make absolutely sure the directory exists.
    path.parent.mkdir(parents=True, exist_ok=True)

    # Write the JSON file.
    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            data,
            f,
            indent=2,
            ensure_ascii=False
        )

    print(f"Camera configuration saved to: {path}")


# Load cameras ONLY from json/cameras.json
cameras = load_cameras()


def _camera_status(message):
    dpg.set_value("camera_manager_status", message)


def _refresh_camera_list():
    """Rebuild the camera list in the Camera Manager tab."""
    if not dpg.does_item_exist("camera_list"):
        return

    dpg.delete_item("camera_list", children_only=True)

    if not cameras.get("camera"):
        dpg.add_text("No cameras configured.", parent="camera_list")
        return

    for camera_id, camera in sorted(cameras["camera"].items(), key=lambda item: int(item[0]) if item[0].isdigit() else item[0]):
        with dpg.group(parent="camera_list", horizontal=True):
            dpg.add_text(
                f"Camera {camera_id} | "
                f"{camera.get('name', '')} | "
                f"{camera.get('camera_url', '')}"
            )
            dpg.add_button(
                label="Delete",
                callback=delete_camera,
                user_data=camera_id,
            )


def _refresh_live_camera_selector():
    """Refresh the Live View camera selector after camera config changes."""
    if not dpg.does_item_exist("live_camera_selector"):
        return

    items = [
        str(camera_id)
        for camera_id in sorted(
            cameras.get("camera", {}).keys(),
            key=lambda value: int(value) if str(value).isdigit() else str(value),
        )
    ]
    dpg.configure_item("live_camera_selector", items=items)

    if items:
        selected = str(_active_camera_id) if _active_camera_id in cameras.get("camera", {}) else items[0]
        dpg.set_value("live_camera_selector", selected)


def add_camera(sender=None, app_data=None):
    """Add a camera to the JSON configuration."""
    camera_id = dpg.get_value("camera_id_input").strip()
    camera_url = dpg.get_value("camera_url_input").strip()
    name = dpg.get_value("camera_name_input").strip()
    location = dpg.get_value("camera_location_input").strip()
    status = dpg.get_value("camera_status_input").strip() or "online"

    if not camera_id:
        _camera_status("Camera ID is required.")
        return

    if not camera_id.isdigit():
        _camera_status("Camera ID must be a number, e.g. 1 or 2.")
        return

    if not camera_url:
        _camera_status("Camera URL is required.")
        return

    if camera_id in cameras["camera"]:
        _camera_status(f"Camera {camera_id} already exists. Use a different ID.")
        return

    if not name:
        name = f"Camera {camera_id}"

    cameras["camera"][camera_id] = {
        "camera_url": camera_url,
        "name": name,
        "location": location,
        "status": status,
    }

    try:
        save_cameras()
    except OSError as e:
        del cameras["camera"][camera_id]
        _camera_status(f"Failed to save {CAMERAS_PATH}: {e}")
        return

    _refresh_camera_list()
    _refresh_live_camera_selector()
    dpg.set_value("camera_id_input", "")
    dpg.set_value("camera_url_input", "")
    dpg.set_value("camera_name_input", "")
    dpg.set_value("camera_location_input", "")
    dpg.set_value("camera_status_input", "online")
    _camera_status(f"Camera {camera_id} added and saved to {CAMERAS_PATH}.")


def delete_camera(sender=None, app_data=None, user_data=None):
    """Delete a camera from the JSON configuration."""
    camera_id = str(user_data)

    if camera_id not in cameras["camera"]:
        _camera_status(f"Camera {camera_id} was not found.")
        return

    del cameras["camera"][camera_id]

    try:
        save_cameras()
    except OSError as e:
        # We cannot reliably reconstruct the deleted object after a failed
        # write, so reload the file-backed state if possible.
        cameras.clear()
        cameras.update(load_cameras())
        _refresh_camera_list()
        _refresh_live_camera_selector()
        _camera_status(f"Failed to save {CAMERAS_PATH}: {e}")
        return

    _refresh_camera_list()
    _refresh_live_camera_selector()

    # If the active camera was deleted, switch to the first remaining camera.
    if str(_active_camera_id) == camera_id:
        new_camera_id = _get_first_camera_id()
        if new_camera_id is not None:
            select_camera(user_data=new_camera_id)
        else:
            dpg.set_value("camera_placeholder", "No cameras configured.")
            dpg.set_value("active_camera_label", "Active: None")

    _camera_status(f"Camera {camera_id} deleted from {CAMERAS_PATH}.")





PROFILE_MATCH_TOLERANCE = 0.7   # lower = stricter matching, fewer false merges
MAX_PROFILES = 40               # number of pre-allocated texture slots
PROFILE_THUMB_SIZE = FACE_THUMB_SIZE
PROFILE_GRID_COLUMNS = GRID_COLUMNS

profiles = []  # list of dicts: {"encoding": np.ndarray, "hits": int}
ai_result_queue = queue.Queue()
def generate_profile_ai(profile_idx, face_crop):
    """Generate the AI overview in the background."""

    print(f"[AI] Starting profile {profile_idx + 1}")

    try:
        result = asyncio.run(generate_ai_overview(face_crop))
        print(f"[AI] Finished profile {profile_idx + 1}: {result}")
    except Exception as e:
        result = f"AI error: {e}"
        print(f"[AI] ERROR: {e}")

    ai_result_queue.put((profile_idx, result))
def _face_to_rgba_float(face_bgr, size):
    """Resize a BGR face crop and convert it to the flattened float32 RGBA
    buffer dearpygui textures expect."""
    face = cv2.resize(face_bgr, (size, size))
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGBA)
    face = face.astype(np.float32) / 255.0
    return face.flatten()


def find_or_create_profile(face_data, face_crop, tolerance=PROFILE_MATCH_TOLERANCE):
    """Match a face embedding against known profiles. Returns the profile
    index (updating its thumbnail/encoding), or None if no slots remain."""

    best_idx = None
    best_distance = None

    for idx, profile in enumerate(profiles):
        distance = face_recognition.face_distance(
            [profile["encoding"]],
            face_data,
        )[0]

        if distance < tolerance and (best_distance is None or distance < best_distance):
            best_idx = idx
            best_distance = distance

    if best_idx is not None:
        profile = profiles[best_idx]
        profile["hits"] += 1
        profile["last_seen"] = time.time()
        profile["last_crop"] = face_crop.copy()
        profile["encoding"] = (
            profile["encoding"] * 0.8 + face_data * 0.2
        )

        dpg.set_value(
            f"profile_texture_{best_idx}",
            _face_to_rgba_float(face_crop, PROFILE_THUMB_SIZE),
        )
        dpg.set_value(
            f"profile_label_{best_idx}",
            f"Profile {best_idx + 1}  ({profile['hits']} sightings)",
        )
        idx = best_idx
    else:
        # No match: create a new profile if a slot is free.
        if len(profiles) >= MAX_PROFILES:
            return None

        idx = len(profiles)
        profiles.append({
            "encoding": face_data.copy(),
            "hits": 1,
            "first_seen": time.time(),
            "last_seen": time.time(),
            "ai_overview": "Generating AI overview...",
        })

        # Generate AI overview in the background so the GUI doesn't freeze.
        threading.Thread(
            target=generate_profile_ai,
            args=(idx, face_crop.copy()),
            daemon=True,
        ).start()

        dpg.configure_item(f"profile_image_{idx}", show=True)
    # Check this profile against the watchlist database and surface it in
    # the Pings tab if it's a close enough match.
    match = match_watchlist(profiles[idx]["encoding"])
    if match is not None:
        entry, distance = match
        update_ping_slot(idx, face_crop, entry, distance)

    return idx



WATCHLIST_PATH = "watchlist.json"
WATCHLIST_MATCH_TOLERANCE = 0.5
MAX_PINGS = MAX_PROFILES
PING_THUMB_SIZE = FACE_THUMB_SIZE
PING_GRID_COLUMNS = GRID_COLUMNS

# Maps a profile index -> the ping slot index it currently occupies.
ping_slot_map = {}

# Maps a ping slot index -> the extra detail data used to populate the
# detail popup when the user clicks a ping thumbnail.
ping_details = {}


def load_watchlist(path=WATCHLIST_PATH):
    """Load the watchlist JSON database into memory. Malformed or
    missing entries are skipped rather than crashing the app."""

    try:
        with open(path, "r") as f:
            raw_entries = json.load(f)
    except FileNotFoundError:
        print(f"watchlist not found at '{path}', starting with an empty list")
        return []
    except json.JSONDecodeError as e:
        print(f"watchlist at '{path}' is not valid JSON: {e}")
        # Treat an empty/invalid watchlist as an empty database.
        return []

    entries = []

    for i, item in enumerate(raw_entries):
        try:
            embedding = np.array(item["embedding"], dtype=np.float64)
        except (KeyError, TypeError, ValueError) as e:
            print(f"skipping watchlist entry {i}: bad embedding ({e})")
            continue

        if embedding.shape != (128,):
            print(f"skipping watchlist entry {i}: expected 128 values, got {embedding.shape}")
            continue

        entries.append({
            "name": item.get("name", "Unknown"),
            "height": item.get("height"),
            "description": item.get("description", ""),
            "embedding": embedding,
        })

    print(f"loaded {len(entries)} watchlist entries from '{path}'")
    return entries

def delete_watchlist_entry(sender=None, app_data=None, user_data=None):
    """Delete one watchlist entry from memory and watchlist.json."""

    index = int(user_data)

    if index < 0 or index >= len(watchlist):
        return

    entry = watchlist[index]
    name = entry.get("name", "Unknown")

    # Remove from memory
    watchlist.pop(index)

    try:
        # Save the updated list to JSON
        save_watchlist()
    except OSError as e:
        print(f"Failed to save watchlist: {e}")

        # Put the entry back if saving failed
        watchlist.insert(index, entry)

        return

    print(f"Deleted watchlist entry: {name}")

    # Rebuild the visible list
    refresh_watchlist_list()
watchlist = load_watchlist()
def refresh_watchlist_list():
    """Refresh the list of all entries currently stored in watchlist.json."""

    if not dpg.does_item_exist("watchlist_entries"):
        return

    dpg.delete_item("watchlist_entries", children_only=True)

    if not watchlist:
        dpg.add_text(
            "No watchlist entries.",
            parent="watchlist_entries"
        )
        return

    for i, entry in enumerate(watchlist):
        name = entry.get("name", "Unknown")
        height = entry.get("height")
        description = entry.get("description", "")

        with dpg.group(
            parent="watchlist_entries",
            horizontal=True
        ):
            with dpg.group():
                dpg.add_text(
                    f"{i + 1}. {name}"
                )

                if height:
                    dpg.add_text(
                        f"Height: {height}",
                        color=(150, 150, 150),
                    )

                if description:
                    dpg.add_text(
                        description,
                        color=(150, 150, 150),
                        wrap=300,
                    )

            dpg.add_button(
                label="Delete",
                callback=delete_watchlist_entry,
                user_data=i,
            )

        dpg.add_separator(
            parent="watchlist_entries"
        )

def save_watchlist(path=WATCHLIST_PATH):
    """Persist the in-memory watchlist (including any new enrollments)
    back to disk as JSON."""

    serializable = []
    for entry in watchlist:
        embedding = entry["embedding"]
        if isinstance(embedding, np.ndarray):
            embedding = embedding.tolist()

        serializable.append({
            "name": entry["name"],
            "height": entry.get("height"),
            "description": entry.get("description", ""),
            "embedding": embedding,
        })

    with open(path, "w") as f:
        json.dump(serializable, f, indent=2)


def match_watchlist(encoding, tolerance=WATCHLIST_MATCH_TOLERANCE):
    """Return (entry, distance) for the closest watchlist entry within
    tolerance, or None if nothing matches closely enough."""

    if not watchlist:
        return None

    best_entry = None
    best_distance = None

    for entry in watchlist:
        distance = face_recognition.face_distance(
            [entry["embedding"]],
            encoding,
        )[0]

        if distance < tolerance and (best_distance is None or distance < best_distance):
            best_entry = entry
            best_distance = distance

    if best_entry is None:
        return None

    return best_entry, best_distance


def update_ping_slot(profile_idx, face_crop, entry, distance):
    """Show/refresh the Pings tab slot for a profile that matched the
    watchlist. Each profile keeps a stable slot for as long as it keeps
    matching, so it doesn't jump around the grid."""

    if profile_idx not in ping_slot_map:
        if len(ping_slot_map) >= MAX_PINGS:
            return
        ping_slot_map[profile_idx] = len(ping_slot_map)
        dpg.configure_item(f"ping_image_{ping_slot_map[profile_idx]}", show=True)

    slot = ping_slot_map[profile_idx]

    dpg.set_value(
        f"ping_texture_{slot}",
        _face_to_rgba_float(face_crop, PING_THUMB_SIZE),
    )

    confidence = max(0.0, 1.0 - distance)
    height_part = f", {entry['height']}" if entry.get("height") is not None else ""
    dpg.set_value(
        f"ping_label_{slot}",
        f"{entry['name']}{height_part}  ({confidence:.0%} match)",
    )

    now = time.time()
    details = ping_details.get(slot)
    if details is None:
        details = {
            "entry": entry,
            "distance": distance,
            "sightings": 1,
            "first_seen": now,
            "last_seen": now,
            "crop": face_crop.copy(),
        }
        ping_details[slot] = details
    else:
        details["entry"] = entry
        details["distance"] = distance
        details["sightings"] += 1
        details["last_seen"] = now
        details["crop"] = face_crop.copy()

    # Keep the detail popup live if it's currently showing this slot.
    if dpg.does_item_exist("ping_detail_window") and dpg.is_item_shown("ping_detail_window"):
        if dpg.get_item_user_data("ping_detail_window") == slot:
            _refresh_ping_detail(slot)


def _format_ago(seconds_ago):
    if seconds_ago < 60:
        return "just now" if seconds_ago < 5 else f"{int(seconds_ago)}s ago"
    if seconds_ago < 3600:
        return f"{int(seconds_ago // 60)}m ago"
    if seconds_ago < 86400:
        return f"{int(seconds_ago // 3600)}h ago"
    return f"{int(seconds_ago // 86400)}d ago"


def _build_ping_summary(details):
    """Compose a short, plain-language summary line from the stored
    detail data for a ping slot."""

    entry = details["entry"]
    now = time.time()

    confidence = max(0.0, 1.0 - details["distance"])
    sightings = details["sightings"]
    first_ago = _format_ago(now - details["first_seen"])
    last_ago = _format_ago(now - details["last_seen"])

    name = entry.get("name", "Unknown")
    height_bit = f", height {entry['height']}" if entry.get("height") is not None else ""

    if sightings == 1:
        timing_bit = f"Matched once, {last_ago}."
    else:
        timing_bit = (
            f"Matched {sightings} times, first {first_ago} and most recently {last_ago}."
        )

    return (
        f"{name}{height_bit}. "
        f"{timing_bit} "
        f"Current match confidence: {confidence:.0%}."
    )


def _refresh_ping_detail(slot):
    """Populate the detail popup's widgets from ping_details[slot]."""

    details = ping_details.get(slot)
    if details is None:
        return

    entry = details["entry"]
    confidence = max(0.0, 1.0 - details["distance"])

    dpg.set_value(
        "ping_detail_texture",
        _face_to_rgba_float(details["crop"], PING_DETAIL_THUMB_SIZE),
    )
    dpg.set_value("ping_detail_name", entry.get("name", "Unknown"))
    dpg.set_value(
        "ping_detail_height",
        f"Height: {entry['height']}" if entry.get("height") is not None else "Height: (not recorded)",
    )
    dpg.set_value("ping_detail_confidence", f"Match confidence: {confidence:.0%}")
    dpg.set_value("ping_detail_sightings", f"Sightings: {details['sightings']}")
    dpg.set_value(
        "ping_detail_first_seen",
        f"First seen: {_format_ago(time.time() - details['first_seen'])}",
    )
    dpg.set_value(
        "ping_detail_last_seen",
        f"Last seen: {_format_ago(time.time() - details['last_seen'])}",
    )
    dpg.set_value("ping_detail_summary", _build_ping_summary(details))

    dpg.set_item_user_data("ping_detail_window", slot)

    description = entry.get("description", "")

    dpg.set_value(
        "ping_detail_description",
        f"Description: {description}" if description else "Description: (none)",
    )
def show_ping_detail(sender, app_data, user_data):
    """Click handler for a ping thumbnail: opens the detail popup for
    that ping slot, if it has data."""

    slot = user_data
    if slot not in ping_details:
        return

    _refresh_ping_detail(slot)
    dpg.configure_item("ping_detail_window", show=True)


# ============================================================
# Create context
# ============================================================

dpg.create_context()


# ============================================================
# THREADING
# ============================================================


frame_queue = queue.Queue(maxsize=1)
recognition_queue = queue.Queue(maxsize=1)

# Recognition results, updated by the background worker, read by update_camera.
# Each entry: ((x, y, w, h), crop, face_data)
latest_faces = []
faces_lock = threading.Lock()

history_faces_lock = threading.Lock()


def is_new_face(face_data, tolerance=0.5):
    """Vectorized nearest-neighbor check against face history.

    Replaces the old per-item loop (one face_distance call per
    history entry) with a single batched call.
    """
    with history_faces_lock:
        if not history_faces:
            history_faces.append(face_data.copy())
            return True

        distances = face_recognition.face_distance(
            np.array(history_faces),
            face_data
        )

        if distances.min() < tolerance:
            return False

        history_faces.append(face_data.copy())

    return True


def add_history_face(face_data):
    dpg.add_text(
        str(face_data),
        parent="history_list"
    )

    # Cap history so the UI doesn't accumulate unbounded widgets
    # over a long-running session.
    max_history_items = 500
    children = dpg.get_item_children("history_list", 1)
    if children and len(children) > max_history_items:
        for old_item in children[:len(children) - max_history_items]:
            dpg.delete_item(old_item)


# ============================================================
# ACTIVE CAMERA / SINGLE-STREAM CONTROL
# ============================================================

_camera_lock = threading.Lock()
_camera_switch_event = threading.Event()
_camera_thread_stop = threading.Event()
_active_camera_id = None
_active_stream = None


def _get_first_camera_id():
    camera_entries = cameras.get("camera", {})
    if not camera_entries:
        return None
    return sorted(
        camera_entries.keys(),
        key=lambda value: int(value) if str(value).isdigit() else str(value),
    )[0]


def _get_camera_url(camera_id):
    entry = cameras.get("camera", {}).get(str(camera_id))
    if not entry:
        return None
    return entry.get("camera_url")


def _close_stream(stream):
    """Best-effort shutdown for whichever stream implementation is installed."""
    if stream is None:
        return

    for method_name in ("close", "stop", "release"):
        method = getattr(stream, method_name, None)
        if callable(method):
            try:
                method()
            except Exception:
                pass
            break


def _clear_camera_queues():
    """Remove stale frames/results when changing cameras."""
    while True:
        try:
            frame_queue.get_nowait()
        except queue.Empty:
            break

    while True:
        try:
            recognition_queue.get_nowait()
        except queue.Empty:
            break

    with faces_lock:
        latest_faces.clear()


def select_camera(sender=None, app_data=None, user_data=None):
    """Switch the single active camera stream."""
    global _active_camera_id

    camera_id = str(app_data if app_data is not None else user_data)
    if camera_id not in cameras.get("camera", {}):
        return

    with _camera_lock:
        _active_camera_id = camera_id

    _clear_camera_queues()
    _camera_switch_event.set()

    # Ask the current stream to release immediately if the implementation
    # provides close()/stop()/release(). This prevents two cameras from
    # remaining connected during a switch.
    _close_stream(_active_stream)

    camera = cameras["camera"][camera_id]
    dpg.set_value(
        "camera_placeholder",
        f"Connecting to {camera.get('name', f'Camera {camera_id}')}...",
    )
    dpg.set_value("active_camera_label", f"Active: {camera.get('name', f'Camera {camera_id}')}")


def camera_worker():
    """Capture exactly ONE camera stream at a time.

    Selecting another camera signals this worker to stop consuming the
    current stream, close it when possible, and connect to the newly
    selected URL. The recognition worker therefore receives frames from
    only the currently selected camera.
    """
    global _active_stream

    while not _camera_thread_stop.is_set():
        with _camera_lock:
            camera_id = _active_camera_id

        url = _get_camera_url(camera_id)
        if not url:
            time.sleep(0.1)
            continue

        _camera_switch_event.clear()

        try:
            stream = BlenderStream(url)
            _active_stream = stream
            print(f"Streaming Camera {camera_id}: {url}")

            for frame in stream.frames():
                if _camera_thread_stop.is_set() or _camera_switch_event.is_set():
                    break

                if frame is None or getattr(frame, "size", 0) == 0:
                    continue

                if frame_queue.full():
                    try:
                        frame_queue.get_nowait()
                    except queue.Empty:
                        pass

                frame_queue.put(frame)

                # Only the active camera's frames enter recognition.
                if recognition_queue.full():
                    try:
                        recognition_queue.get_nowait()
                    except queue.Empty:
                        pass

                recognition_queue.put(frame)

        except Exception as e:
            print(f"Camera {camera_id} stream error: {e}")
            print(f"Camera {camera_id} error: {e}")
            time.sleep(0.5)
        finally:
            _close_stream(stream if 'stream' in locals() else None)
            _active_stream = None

        # If a switch occurred, immediately connect to the new camera.
        # Otherwise retry the same camera after a short pause if its stream
        # ended unexpectedly.
        if _camera_switch_event.is_set():
            continue

        time.sleep(0.25)


def recognition_worker():
    """Runs face detection + encoding off the render thread.

    This is the main fix for the "clanky" feel: face_recognition
    encoding is the slowest part of the pipeline, and previously it
    ran synchronously inside update_camera() on the same thread that
    calls dpg.render_dearpygui_frame(). Moving it here means the UI
    thread never blocks on it -- it just reads whatever the most
    recent result is.
    """
    while True:
        frame = recognition_queue.get()

        if frame is None or frame.size == 0:
            continue

        faces = detect_faces(frame)
        results = []

        for face in faces:

            box = face["box"]
            crop = face["crop"]
            face_location = face["face_location"]

            if crop is None or crop.size == 0:
                continue

            face_data = get_face_data(
                crop,
                face_location,
            )

            results.append(
                (
                    box,
                    crop,
                    face_data,
                )
            )
        with faces_lock:
            latest_faces[:] = results

# ============================================================
# BLENDER STREAM
# ============================================================

CAMERA_WIDTH = 860
CAMERA_HEIGHT = 620

_active_camera_id = _get_first_camera_id()

camera_thread = threading.Thread(target=camera_worker, daemon=True)
camera_thread.start()

recognition_thread = threading.Thread(target=recognition_worker, daemon=True)
recognition_thread.start()

with dpg.texture_registry():
    dpg.add_raw_texture(
        width=CAMERA_WIDTH,
        height=CAMERA_HEIGHT,
        default_value=np.zeros(CAMERA_WIDTH * CAMERA_HEIGHT * 4, dtype=np.float32),
        format=dpg.mvFormat_Float_rgba,
        tag="camera_texture"
    )
with dpg.texture_registry():

    for i in range(MAX_FACE_SLOTS):

        dpg.add_raw_texture(
            width=FACE_THUMB_SIZE,
            height=FACE_THUMB_SIZE,
            default_value=np.zeros(
                FACE_THUMB_SIZE * FACE_THUMB_SIZE * 4,
                dtype=np.float32
            ),
            format=dpg.mvFormat_Float_rgba,
            tag=f"face_texture_{i}"
        )

with dpg.texture_registry():

    for i in range(MAX_PROFILES):

        dpg.add_raw_texture(
            width=PROFILE_THUMB_SIZE,
            height=PROFILE_THUMB_SIZE,
            default_value=np.zeros(
                PROFILE_THUMB_SIZE * PROFILE_THUMB_SIZE * 4,
                dtype=np.float32
            ),
            format=dpg.mvFormat_Float_rgba,
            tag=f"profile_texture_{i}"
        )

with dpg.texture_registry():

    for i in range(MAX_PINGS):

        dpg.add_raw_texture(
            width=PING_THUMB_SIZE,
            height=PING_THUMB_SIZE,
            default_value=np.zeros(
                PING_THUMB_SIZE * PING_THUMB_SIZE * 4,
                dtype=np.float32
            ),
            format=dpg.mvFormat_Float_rgba,
            tag=f"ping_texture_{i}"
        )

ADDDATA_THUMB_SIZE = FACE_THUMB_SIZE * 2  # a bit bigger so it's easier to check

with dpg.texture_registry():
    dpg.add_raw_texture(
        width=ADDDATA_THUMB_SIZE,
        height=ADDDATA_THUMB_SIZE,
        default_value=np.zeros(
            ADDDATA_THUMB_SIZE * ADDDATA_THUMB_SIZE * 4,
            dtype=np.float32
        ),
        format=dpg.mvFormat_Float_rgba,
        tag="adddata_preview_texture"
    )

PING_DETAIL_THUMB_SIZE = FACE_THUMB_SIZE * 3

with dpg.texture_registry():
    dpg.add_raw_texture(
        width=PING_DETAIL_THUMB_SIZE,
        height=PING_DETAIL_THUMB_SIZE,
        default_value=np.zeros(
            PING_DETAIL_THUMB_SIZE * PING_DETAIL_THUMB_SIZE * 4,
            dtype=np.float32
        ),
        format=dpg.mvFormat_Float_rgba,
        tag="ping_detail_texture"
    )
with dpg.texture_registry():
    dpg.add_raw_texture(
        width=PROFILE_THUMB_SIZE * 2,
        height=PROFILE_THUMB_SIZE * 2,
        default_value=np.zeros(
            (PROFILE_THUMB_SIZE * 2) *
            (PROFILE_THUMB_SIZE * 2) *
            4,
            dtype=np.float32
        ),
        format=dpg.mvFormat_Float_rgba,
        tag="profile_detail_texture"
    )
def get_face_data(face_crop, face_location):
    if face_crop is None or face_crop.size == 0:
        return None

    face_rgb = cv2.cvtColor(
        face_crop,
        cv2.COLOR_BGR2RGB
    )

    h, w = face_rgb.shape[:2]

    top, right, bottom, left = face_location

    # Clamp coordinates to the crop
    top = max(0, min(int(top), h - 1))
    bottom = max(top + 1, min(int(bottom), h))

    left = max(0, min(int(left), w - 1))
    right = max(left + 1, min(int(right), w))

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


# ============================================================
# ADD DATA (watchlist enrollment)
# ============================================================

# Holds whatever face has been captured/loaded but not yet saved.
_pending_enrollment = {"embedding": None, "crop": None}


def _set_adddata_status(message):
    dpg.set_value("adddata_status", message)


def capture_face_from_live(sender=None, app_data=None):
    """Grab the first currently-detected face from the live feed as the
    pending enrollment capture."""

    with faces_lock:
        faces = list(latest_faces)

    if not faces:
        _set_adddata_status("No face currently detected in the live feed.")
        return

    box, crop, face_data = faces[0]

    if face_data is None or crop is None or crop.size == 0:
        _set_adddata_status("Could not compute an embedding for the detected face.")
        return

    _pending_enrollment["embedding"] = face_data.copy()
    _pending_enrollment["crop"] = crop.copy()

    dpg.set_value(
        "adddata_preview_texture",
        _face_to_rgba_float(crop, ADDDATA_THUMB_SIZE),
    )
    dpg.configure_item("adddata_preview_image", show=True)
    _set_adddata_status("Captured face from the live feed. Fill in the details and save.")


def load_face_from_file(sender, app_data):
    """File dialog callback: detect the largest face in the chosen image
    and use it as the pending enrollment capture."""

    file_path = app_data.get("file_path_name") if app_data else None
    if not file_path:
        return

    image = cv2.imread(file_path)
    if image is None:
        _set_adddata_status(f"Could not read image: {file_path}")
        return

    faces = detect_faces(image)
    if faces is None or len(faces) == 0:
        _set_adddata_status("No face found in that image.")
        return

    # If more than one face was detected, use the largest box.
    x, y, w_box, h_box = max(faces, key=lambda b: b[2] * b[3])
    crop = image[y:y + h_box, x:x + w_box]

    if crop.size == 0:
        _set_adddata_status("Detected face region was empty.")
        return

    face_data = get_face_data(crop)
    if face_data is None:
        _set_adddata_status("Could not compute an embedding for that face.")
        return

    _pending_enrollment["embedding"] = face_data
    _pending_enrollment["crop"] = crop.copy()

    dpg.set_value(
        "adddata_preview_texture",
        _face_to_rgba_float(crop, ADDDATA_THUMB_SIZE),
    )
    dpg.configure_item("adddata_preview_image", show=True)
    _set_adddata_status(f"Loaded face from {file_path}. Fill in the details and save.")


def save_enrollment(sender=None, app_data=None):
    """Append the pending capture + form details to the watchlist and
    persist it to WATCHLIST_PATH."""

    embedding = _pending_enrollment["embedding"]

    if embedding is None:
        _set_adddata_status("Capture a face from the live feed or load one from a file first.")
        return

    name = dpg.get_value("adddata_name").strip()
    if not name:
        _set_adddata_status("Name is required.")
        return

    height_raw = dpg.get_value("adddata_height").strip()
    height = height_raw if height_raw else None
    description = dpg.get_value("adddata_description").strip()

    watchlist.append({
        "name": name,
        "height": height,
        "description": description,
        "embedding": embedding,
    })

    try:
        save_watchlist()
    except OSError as e:
        # Roll back the in-memory add so it stays consistent with disk.
        watchlist.pop()
        _set_adddata_status(f"Failed to write {WATCHLIST_PATH}: {e}")
        return

    _set_adddata_status(f"Saved '{name}' to {WATCHLIST_PATH} ({len(watchlist)} entries total).")

    # Reset the form for the next entry.
    dpg.set_value("adddata_name", "")
    dpg.set_value("adddata_height", "")
    _pending_enrollment["embedding"] = None
    _pending_enrollment["crop"] = None
    dpg.configure_item("adddata_preview_image", show=False)


# Preallocated buffers, reused every frame instead of being
# reallocated with np.zeros()/astype() on every call.
_canvas = np.zeros((CAMERA_HEIGHT, CAMERA_WIDTH, 4), dtype=np.uint8)
_canvas_f32 = np.empty((CAMERA_HEIGHT, CAMERA_WIDTH, 4), dtype=np.float32)
_prev_visible_count = -1

def process_ai_results():
    while True:
        try:
            profile_idx, result = ai_result_queue.get_nowait()
        except queue.Empty:
            break

        if profile_idx >= len(profiles):
            continue

        # Store the result
        profiles[profile_idx]["ai_overview"] = result

        print(
            f"[AI] GUI updated for Profile {profile_idx + 1}: {result}"
        )

        # If this profile window is currently open, update it immediately
        if (
            dpg.does_item_exist("profile_detail_window")
            and dpg.is_item_shown("profile_detail_window")
        ):
            dpg.set_value(
                "profile_detail_ai",
                result
            )
def update_camera():
    global _prev_visible_count

    try:
        frame = frame_queue.get_nowait()
    except queue.Empty:
        return

    if frame is None or frame.size == 0:
        return

    h, w = frame.shape[:2]

    if w == 0 or h == 0:
        return

    # Pull whatever the recognition worker has most recently produced.
    # This never blocks the render loop on face_recognition's cost.
    with faces_lock:
        faces = list(latest_faces)

    # Show/hide face thumbnails only when the visible count changes,
    # instead of calling configure_item MAX_FACE_SLOTS times every frame.
    if _prev_visible_count != len(faces):
        for i in range(MAX_FACE_SLOTS):
            dpg.configure_item(
                f"face_image_{i}",
                show=i < len(faces)
            )
        _prev_visible_count = len(faces)

    for i, (box, face, face_data) in enumerate(faces):
        x, y, w_box, h_box = box

        cv2.rectangle(frame, (x, y), (x + w_box, y + h_box), (0, 255, 0), 2)

        if i >= MAX_FACE_SLOTS:
            continue

        if face.size == 0:
            continue

        if face_data is not None:
            if is_new_face(face_data):
                add_history_face(face_data)

            # Match/create a profile for this embedding and refresh its
            # thumbnail in the Profiles tab.
            find_or_create_profile(face_data, face)

        dpg.set_value(
            f"face_texture_{i}",
            _face_to_rgba_float(face, FACE_THUMB_SIZE),
        )

    # Preserve aspect ratio
    scale = min(CAMERA_WIDTH / w, CAMERA_HEIGHT / h)
    new_width = max(int(w * scale), 1)
    new_height = max(int(h * scale), 1)

    try:
        resized = cv2.resize(frame, (new_width, new_height))
    except cv2.error as e:
        print("resize failed:", e)
        return

    resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGBA)

    _canvas[:] = 0

    x_offset = (CAMERA_WIDTH - new_width) // 2
    y_offset = (CAMERA_HEIGHT - new_height) // 2

    _canvas[y_offset:y_offset + resized.shape[0], x_offset:x_offset + resized.shape[1]] = resized

    np.divide(_canvas, 255.0, out=_canvas_f32, casting="unsafe")
    flat = _canvas_f32.ravel()

    expected_size = CAMERA_WIDTH * CAMERA_HEIGHT * 4

    if flat.size != expected_size:
        print(f"texture size mismatch: got {flat.size}, expected {expected_size}")
        return

    dpg.set_value("camera_texture", flat)

# ============================================================
# Main application window
# ============================================================
def show_profile(sender, app_data, user_data):
    idx = user_data

    if idx >= len(profiles):
        return

    profile = profiles[idx]

    dpg.set_value(
        "profile_detail_title",
        f"Profile {idx + 1}"
    )

    dpg.set_value(
        "profile_detail_sightings",
        f"Sightings: {profile['hits']}"
    )

    dpg.set_value(
        "profile_detail_first_seen",
        f"First seen: {time.strftime('%d/%m/%Y %H:%M:%S', time.localtime(profile.get('first_seen')))}"
    )

    dpg.set_value(
        "profile_detail_last_seen",
        f"Last seen: {time.strftime('%d/%m/%Y %H:%M:%S', time.localtime(profile.get('last_seen')))}"
    )

    dpg.set_value(
        "profile_detail_embedding",
        "128 dimensions"
    )

    dpg.set_value(
        "profile_detail_ai",
        profile.get("ai_overview") or "Not generated yet."
    )

    # Show the latest captured face
    if "last_crop" in profile and profile["last_crop"] is not None:
        dpg.set_value(
            "profile_detail_texture",
            _face_to_rgba_float(
                profile["last_crop"],
                PROFILE_THUMB_SIZE * 2
            )
        )

    dpg.configure_item(
        "profile_detail_window",
        show=True
    )
with dpg.window(
    label="Surveillance System", tag="main_window", no_collapse=True,):

    with dpg.tab_bar():

        # ----------------------------------------------------
        # Live View
        # ----------------------------------------------------
        with dpg.tab(label="Live View"):

            # Only the selected camera is connected and scanned.
            dpg.add_text("CAMERA SELECTION")
            with dpg.group(horizontal=True):
                camera_items = {}
                for camera_id, camera in sorted(
                    cameras.get("camera", {}).items(),
                    key=lambda item: int(item[0]) if item[0].isdigit() else item[0],
                ):
                    camera_items[str(camera_id)] = camera.get("name", f"Camera {camera_id}")

                dpg.add_combo(
                    items=list(camera_items.keys()),
                    default_value=_active_camera_id or "",
                    tag="live_camera_selector",
                    callback=select_camera,
                    width=220,
                )
                dpg.add_text(
                    f"Active: {camera_items.get(_active_camera_id, 'None')}",
                    tag="active_camera_label",
                )

            dpg.add_separator()

            with dpg.group(horizontal=True):

                # LEFT SIDE — CAMERA
                with dpg.child_window(width=900, height=700, border=True):
                    dpg.add_text("LIVE CAMERA")
                    dpg.add_separator()

                    dpg.add_image("camera_texture", width=CAMERA_WIDTH, height=CAMERA_HEIGHT)

                    dpg.add_text("Camera feed will go here", tag="camera_placeholder")

                # RIGHT SIDE — INFORMATION
                with dpg.child_window(
                    width=450,
                    height=700,
                    border=True
                ):

                    dpg.add_text("DETECTED FACES")
                    dpg.add_separator()

                    dpg.add_text(
                        "Live faces",
                        tag="face_count"
                    )
                    with dpg.group(horizontal=True):

                        for i in range(MAX_FACE_SLOTS):

                            dpg.add_image(
                                f"face_texture_{i}",
                                width=FACE_THUMB_SIZE,
                                height=FACE_THUMB_SIZE,
                                tag=f"face_image_{i}"
                            )

        # ----------------------------------------------------
        # Profiles
        # ----------------------------------------------------
        with dpg.tab(label="Profiles"):
            dpg.add_text("FACE PROFILES")
            dpg.add_text(
                "Faces with similar embeddings are grouped into the same profile.",
                color=(150, 150, 150),
            )
            dpg.add_separator()

            with dpg.child_window(width=-1, height=-1, border=True):

                with dpg.group(tag="profiles_grid"):

                    # Pre-built grid, hidden until a profile is created.
                    row_group = None

                    for i in range(MAX_PROFILES):

                        if i % PROFILE_GRID_COLUMNS == 0:
                            row_group = dpg.add_group(horizontal=True)

                        with dpg.group(parent=row_group):

                            dpg.add_image(
                                f"profile_texture_{i}",
                                width=PROFILE_THUMB_SIZE,
                                height=PROFILE_THUMB_SIZE,
                                tag=f"profile_image_{i}",
                                show=False,
                            )
                            with dpg.item_handler_registry(tag=f"profile_handler_{i}"):
                                dpg.add_item_clicked_handler(
                                    callback=show_profile,
                                    user_data=i,
                                )

                            dpg.bind_item_handler_registry(
                                f"profile_image_{i}",
                                f"profile_handler_{i}",
                            )
                            dpg.add_text(
                                "",
                                tag=f"profile_label_{i}",
                            )

        # ----------------------------------------------------
        # Pings
        # ----------------------------------------------------
        with dpg.tab(label="Pings"):
            dpg.add_text("WATCHLIST PINGS")
            dpg.add_text(
                f"Profiles matching an entry in {WATCHLIST_PATH} are shown here.",
                color=(150, 150, 150),
            )
            dpg.add_separator()

            with dpg.child_window(width=-1, height=-1, border=True):

                with dpg.group(tag="pings_grid"):

                    row_group = None

                    for i in range(MAX_PINGS):

                        if i % PING_GRID_COLUMNS == 0:
                            row_group = dpg.add_group(horizontal=True)

                        with dpg.group(parent=row_group):

                            dpg.add_image(
                                f"ping_texture_{i}",
                                width=PING_THUMB_SIZE,
                                height=PING_THUMB_SIZE,
                                tag=f"ping_image_{i}",
                                show=False,
                            )
                            dpg.add_text(
                                "",
                                tag=f"ping_label_{i}",
                            )

                            # Clicking a ping thumbnail opens the detail
                            # popup for that slot.
                            with dpg.item_handler_registry(tag=f"ping_handler_{i}"):
                                dpg.add_item_clicked_handler(
                                    callback=show_ping_detail,
                                    user_data=i,
                                )
                            dpg.bind_item_handler_registry(
                                f"ping_image_{i}", f"ping_handler_{i}"
                            )

        # ----------------------------------------------------
        # Add Data
        # ----------------------------------------------------
        with dpg.tab(label="Data"):
            dpg.add_text("ENROLL A NEW WATCHLIST ENTRY")
            dpg.add_text(
                f"Captures a face embedding and saves it, with the details below, to {WATCHLIST_PATH}.",
                color=(150, 150, 150),
            )
            dpg.add_separator()

            with dpg.group(horizontal=True):

                # LEFT — capture
                with dpg.child_window(width=340, height=420, border=True):
                    dpg.add_text("1. Get a face")
                    dpg.add_separator()

                    dpg.add_image(
                        "adddata_preview_texture",
                        width=ADDDATA_THUMB_SIZE,
                        height=ADDDATA_THUMB_SIZE,
                        tag="adddata_preview_image",
                        show=False,
                    )

                    dpg.add_button(
                        label="Capture from live feed",
                        callback=capture_face_from_live,
                        width=-1,
                    )
                    dpg.add_button(
                        label="Load from image file...",
                        callback=lambda: dpg.show_item("adddata_file_dialog"),
                        width=-1,
                    )

                # RIGHT — details + save
                with dpg.child_window(width=380, height=420, border=True):
                    dpg.add_text("2. Enter details")
                    dpg.add_separator()

                    dpg.add_input_text(label="Name", tag="adddata_name")
                    dpg.add_input_text(
                        label="Height",
                        tag="adddata_height",
                        hint="optional, e.g. 5'10\"",
                    )
                    dpg.add_input_text(
                        label="Description",
                        tag="adddata_description",
                        hint="optional",
                        multiline=True,
                        height=100,
)

                    dpg.add_separator()
                    dpg.add_button(
                        label="Save to watchlist",
                        callback=save_enrollment,
                        width=-1,
                    )
                    dpg.add_text("", tag="adddata_status", wrap=360, color=(150, 220, 150))
            dpg.add_separator()

            dpg.add_text("CURRENT WATCHLIST")
            dpg.add_text(
                "All entries currently stored in watchlist.json.",
                color=(150, 150, 150),
            )

            with dpg.child_window(
                tag="watchlist_entries",
                width=-1,
                height=250,
                border=True,
            ):
                pass

            dpg.add_button(
                label="Refresh Watchlist",
                callback=refresh_watchlist_list,
                width=-1,
            )

        # ----------------------------------------------------
        # History
        # ----------------------------------------------------
        with dpg.tab(label="History"):

            dpg.add_text("DETECTION HISTORY")
            dpg.add_separator()

            with dpg.child_window(
                tag="history_list",
                width=-1,
                height=-1
            ):
                dpg.add_text(
                    "Recognition events will appear here.",
                    tag="history_placeholder"
                )
        # ----------------------------------------------------
        # Camera Manager
        # ----------------------------------------------------
        with dpg.tab(label="Camera Manager"):
            dpg.add_text("CAMERA CONFIGURATION")
            dpg.add_text(
                f"Add or delete cameras stored in {CAMERAS_PATH}.",
                color=(150, 150, 150),
            )
            dpg.add_separator()

            with dpg.group(horizontal=True):
                with dpg.child_window(width=420, height=430, border=True):
                    dpg.add_text("ADD CAMERA")
                    dpg.add_separator()

                    dpg.add_input_text(
                        label="Camera ID",
                        tag="camera_id_input",
                        hint="e.g. 3",
                    )
                    dpg.add_input_text(
                        label="Camera URL",
                        tag="camera_url_input",
                        hint="http://192.168.0.x:port/stream",
                    )
                    dpg.add_input_text(
                        label="Name",
                        tag="camera_name_input",
                        hint="e.g. Camera 3",
                    )
                    dpg.add_input_text(
                        label="Location",
                        tag="camera_location_input",
                        hint="optional",
                    )
                    dpg.add_input_text(
                        label="Status",
                        tag="camera_status_input",
                        default_value="online",
                    )

                    dpg.add_separator()
                    dpg.add_button(
                        label="Add Camera",
                        callback=add_camera,
                        width=-1,
                    )
                    dpg.add_text(
                        "",
                        tag="camera_manager_status",
                        wrap=380,
                        color=(150, 220, 150),
                    )

                with dpg.child_window(width=-1, height=430, border=True):
                    dpg.add_text("CONFIGURED CAMERAS")
                    dpg.add_separator()

                    with dpg.child_window(
                        tag="camera_list",
                        width=-1,
                        height=350,
                        border=False,
                    ):
                        pass

                    dpg.add_button(
                        label="Refresh List",
                        callback=_refresh_camera_list,
                        width=-1,
                    )

            _refresh_camera_list()

        # ----------------------------------------------------
        # Settings
        # ----------------------------------------------------
        with dpg.tab(label="Settings"):
            dpg.add_text("SYSTEM SETTINGS")
            dpg.add_separator()
            dpg.add_text("Settings will go here.")

# Detail popup shown when a Pings thumbnail is clicked. Built once and
# re-populated in place by _refresh_ping_detail() / show_ping_detail().

with dpg.window(
    label="Ping Details",
    tag="ping_detail_window",
    show=False,
    modal=True,
    no_collapse=True,
    width=420,
    height=420,
    pos=(480, 240),
):
    
    dpg.add_image(
        "ping_detail_texture",
        width=PING_DETAIL_THUMB_SIZE,
        height=PING_DETAIL_THUMB_SIZE,
        tag="ping_detail_image",
    )

    dpg.add_separator()
    
    dpg.add_text("", tag="ping_detail_name")
    dpg.add_text("", tag="ping_detail_height")
    dpg.add_text("", tag="ping_detail_confidence")
    dpg.add_text("", tag="ping_detail_sightings")
    dpg.add_text("", tag="ping_detail_first_seen")
    dpg.add_text("", tag="ping_detail_last_seen")
    dpg.add_text("", tag="ping_detail_description")

    dpg.add_separator()

    dpg.add_text(
        "Summary",
        color=(150, 150, 150)
    )

    dpg.add_text(
        "",
        tag="ping_detail_summary",
        wrap=390,
        color=(150, 220, 150),
    )

    dpg.add_separator()

    dpg.add_button(
        label="Close",
        width=-1,
        callback=lambda: dpg.configure_item(
            "ping_detail_window",
            show=False
        ),
    )


with dpg.window(
    label="Profile Details",
    tag="profile_detail_window",
    show=False,
    modal=True,
    no_collapse=True,
    width=450,
    height=500,
):
    dpg.add_text(
    "",
    tag="profile_detail_title"
    )

    dpg.add_image(
        "profile_detail_texture",
        width=PROFILE_THUMB_SIZE * 2,
        height=PROFILE_THUMB_SIZE * 2,
    )


    dpg.add_separator()

    dpg.add_text(
        "",
        tag="profile_detail_sightings"
    )

    dpg.add_text(
        "",
        tag="profile_detail_first_seen"
    )

    dpg.add_text(
        "",
        tag="profile_detail_last_seen"
    )

    dpg.add_separator()

    dpg.add_text(
        "Embedding",
        color=(150, 150, 150)
    )

    dpg.add_text(
        "128 dimensions",
        tag="profile_detail_embedding"
    )

    dpg.add_separator()

    dpg.add_text(
        "AI Overview",
        color=(150, 150, 150)
    )

    dpg.add_text(
        "Not generated yet.",
        tag="profile_detail_ai",
        wrap=400
    )

    dpg.add_separator()

    dpg.add_button(
        label="Close",
        width=-1,
        callback=lambda: dpg.configure_item(
            "profile_detail_window",
            show=False
        ),
    )

# File dialog used by the "Load from image file..." button above.
with dpg.file_dialog(
    directory_selector=False,
    show=False,
    callback=load_face_from_file,
    tag="adddata_file_dialog",
    width=700,
    height=400,
):
    dpg.add_file_extension(".png")
    dpg.add_file_extension(".jpg")
    dpg.add_file_extension(".jpeg")
    dpg.add_file_extension(".*")


# ============================================================
# Viewport
# ============================================================
refresh_watchlist_list()
dpg.create_viewport(
    title="Surveillance System",
    width=1400,
    height=900,
    min_width=1000,
    min_height=700,
)


dpg.setup_dearpygui()

dpg.show_viewport()

dpg.set_primary_window("main_window", True)


# ============================================================
# MAIN LOOP
# ============================================================

while dpg.is_dearpygui_running():

    update_camera()
    #process_ai_results()

    dpg.render_dearpygui_frame()


# ============================================================
# CLEANUP
# ============================================================

_camera_thread_stop.set()
_camera_switch_event.set()
_close_stream(_active_stream)

dpg.destroy_context()