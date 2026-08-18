# ============================================================
# Configuration
# ============================================================

STREAM_URL = "http://192.168.0.41:8000/stream"
WINDOW_TITLE = "Blender Face Detection"

# Fixed width of the small left video box
VIDEO_WIDTH = 480

# Live face thumbnail grid settings (top-right section)
FACE_THUMB_SIZE = 140
GRID_COLUMNS = 4
MAX_FACE_SLOTS = 40
LIVE_SECTION_HEIGHT = 260  # fixed pixel height of the scrollable live area

# Successful Hits table settings (HITS tab)
SAVE_INTERVAL_SECONDS = 2.0    # min time between recorded hits, to avoid flooding the table
HITS_MAX_ROWS = 500            # oldest rows are dropped once this many hits are logged
