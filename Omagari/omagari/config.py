# omagari/config.py
"""
Centralized constants & tunables.

Everything is intentionally plain constants (no imports, no I/O).
Controllers/services read from here only; mutation at runtime is discouraged.
"""

# ---- History / Undo-Redo ----
HISTORY_MAX = 100                 # cap the number of snapshots
HISTORY_SNAPSHOT_ON_DRAG_START = True  # push exactly once when a drag/resize starts

# ---- Navigation timing ----
NAV_INTERVAL_MS = 10              # repeat step interval while holding
NAV_HOLD_MS = 350                 # press duration before repeat kicks in
SLIDER_DEBOUNCE_MS = 50           # delay to coalesce slider drags into a single load

# ---- YOLO defaults ----
YOLO_DEFAULT_CONF = 0.25
YOLO_VERBOSE = False

# ---- Device / Threads defaults ----
CPU_THREADS_DEFAULT = 4           # will be clamped to detected logical max
GPU_COUNT_DEFAULT = 1

# ---- Worker / Async inference ----
WORKER_QUEUE_MAXSIZE = 1          # "latest-wins": keep only the newest request
WORKER_POLL_MS = 30               # UI poll period for completed inference results

# ---- UI sizing ----
RIGHT_PANEL_MIN_WIDTH = 209
INITIAL_WINDOW_GEOMETRY = "1250x800"

# ---- Image / I/O ----
SUPPORTED_IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")

# Save dialog filters (label, pattern)
SAVE_FILETYPES = [
    ("CSV", "*.csv"),
    ("TSV", "*.tsv"),
    ("JSON", "*.json"),
    ("XML", "*.xml"),
    ("All", "*.*"),
]

LOAD_ANNOT_FILETYPES = [
    ("Annotation files", "*.csv *.tsv *.json *.xml *.txt"),
    ("All", "*.*"),
]

# ---- Logging ----
# "DEBUG" | "INFO" | "WARNING" | "ERROR"
LOG_LEVEL = "INFO"

# ---- Selection / drawing ----
DEFAULT_CLASS_NAME = "object"
DEFAULT_CLASS_COLOR = "#00FF00"
SELECTED_BOX_OUTLINE_WIDTH = 2
UNSELECTED_BOX_OUTLINE_WIDTH = 1
FLASH_HIGHLIGHT_WIDTH = 4
HANDLE_RADIUS_PX = 3

# ---- Crosshair ----
# Crosshair logic/drawing must remain byte-for-byte equivalent in view.py
# These flags exist only for nearby behaviors outside the forbidden zone.
CROSSHAIR_ENABLED = True  # do not alter actual crosshair algorithm

# ---- Safety / parsing ----
CSV_TSV_STRICT = True     # skip malformed rows quietly but never crash
XML_STRICT = True
JSON_STRICT = True
YOLO_TXT_STRICT = True

# ---- Misc UX ----
STATUS_READY_TEXT = "Ready"
ABOUT_TITLE = "About"
APP_TITLE = "Omagari"

# ---- Keys (document only; actual binding lives in view/controller) ----
NAV_KEYS = {
    "NEXT": ("Right", "d"),
    "PREV": ("Left", "a"),
}
LABEL_HOTKEYS_DIGITS = tuple(range(0, 10))  # 0..9

# ---- Persistence of last paths (optional; kept in-memory unless you add a settings store) ----
REMEMBER_LAST_SAVE_PATH = True
REMEMBER_LAST_OPEN_DIR = True
