# omagari/domain.py
"""
Domain models & tiny utilities.
No UI, no I/O, no framework imports. Pure data + helpers.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, List, Set, Dict, Tuple, Iterable, Any
import os
import sys
import random

# ---------- Utilities ----------

def clamp(v: float, lo: float, hi: float) -> float:
    """Clamp v into [lo, hi]."""
    if lo > hi:
        lo, hi = hi, lo
    return hi if v > hi else lo if v < lo else v

def normalize_box(x1: float, y1: float, x2: float, y2: float) -> Tuple[float, float, float, float]:
    """Ensure x1<=x2 and y1<=y2."""
    if x1 > x2:
        x1, x2 = x2, x1
    if y1 > y2:
        y1, y2 = y2, y1
    return x1, y1, x2, y2

def random_color(seed: Optional[int] = None) -> str:
    """Generate a readable random color hex."""
    rnd = random.Random(seed)
    r = rnd.randint(32, 255)
    g = rnd.randint(32, 255)
    b = rnd.randint(32, 255)
    return f"#{r:02X}{g:02X}{b:02X}"

def is_number(s: str) -> bool:
    """Lenient numeric check used by parsers."""
    try:
        float(s)
        return True
    except Exception:
        return False

def resource_path(relative_path: str) -> str:
    """
    Resolve resource path for PyInstaller or normal run.
    Keeps backward compatibility with existing code.
    """
    base_path = getattr(sys, "_MEIPASS", None)  # PyInstaller temp dir
    if base_path and os.path.isdir(base_path):
        return os.path.join(base_path, relative_path)
    try:
        here = os.path.dirname(os.path.abspath(__file__))
    except Exception:
        here = os.getcwd()
    return os.path.join(here, relative_path)

# ---------- Data Models ----------

@dataclass
class Box:
    """Axis-aligned bounding box."""
    x1: float
    y1: float
    x2: float
    y2: float
    label: str
    score: Optional[float] = None

    def as_xyxy(self) -> Tuple[int, int, int, int]:
        x1, y1, x2, y2 = normalize_box(self.x1, self.y1, self.x2, self.y2)
        return int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))

    def clamp_to(self, w: int, h: int) -> "Box":
        x1, y1, x2, y2 = self.as_xyxy()
        x1 = clamp(x1, 0, max(0, w - 1))
        x2 = clamp(x2, 0, max(0, w - 1))
        y1 = clamp(y1, 0, max(0, h - 1))
        y2 = clamp(y2, 0, max(0, h - 1))
        x1, y1, x2, y2 = normalize_box(x1, y1, x2, y2)
        return Box(x1, y1, x2, y2, self.label, self.score)

    def moved(self, dx: float, dy: float) -> "Box":
        return Box(self.x1 + dx, self.y1 + dy, self.x2 + dx, self.y2 + dy, self.label, self.score)

    def resized(self, nx1: float, ny1: float, nx2: float, ny2: float) -> "Box":
        x1, y1, x2, y2 = normalize_box(nx1, ny1, nx2, ny2)
        return Box(x1, y1, x2, y2, self.label, self.score)

    def to_list(self) -> List[Any]:
        x1, y1, x2, y2 = self.as_xyxy()
        return [x1, y1, x2, y2, self.label, self.score]

    @staticmethod
    def from_iter(data: Iterable[Any]) -> "Box":
        x1, y1, x2, y2, label, *rest = list(data)
        score = rest[0] if rest else None
        return Box(float(x1), float(y1), float(x2), float(y2), str(label), (None if score is None else float(score)))

# Key types
ImagePath = str
Annotations = Dict[ImagePath, List[Box]]

@dataclass
class AppState:
    """
    Pure state container. No I/O or UI calls.
    Controllers/services mutate this, view renders from it.
    """

    # ---------- YOLO-related state (paths/settings only; 엔진 객체는 서비스가 소유) ----------
    yolo_model_paths: List[str] = field(default_factory=list)
    current_yolo_model_path: Optional[str] = None
    yolo_conf: float = 0.25
    yolo_enabled_classes: Optional[Set[str]] = None  # None = allow all
    yolo_device_mode: str = "CPU"                    # "CPU" | "GPU"
    yolo_cpu_threads: int = 4
    yolo_gpu_count: int = 1
    yolo_auto_predict: bool = False

    # ---------- Files & navigation ----------
    image_files: List[ImagePath] = field(default_factory=list)
    current_index: int = -1
    image_dir: Optional[str] = None

    # ---------- Image sizes (for clamping math) ----------
    img_w: int = 0
    img_h: int = 0

    # ---------- View transform ----------
    scale: float = 1.0
    offset_x: float = 0.0
    offset_y: float = 0.0
    view_initialized: bool = False

    # ---------- Annotations: path -> boxes ----------
    annotations: Annotations = field(default_factory=dict)

    # ---------- Class/label palette ----------
    class_list: List[str] = field(default_factory=lambda: ["object"])
    class_colors: Dict[str, str] = field(default_factory=lambda: {"object": "#00FF00"})
    current_class: str = "object"

    # ---------- Selection ----------
    selected_indices: List[int] = field(default_factory=list)
    active_box: int = -1

    # ---------- Flags (UI toggles mirrored here) ----------
    is_seq_labeling: bool = False
    is_repeating: bool = False

    # ---------- Caches / helpers ----------
    basename_to_path: Dict[str, str] = field(default_factory=dict)     # "IMG_0001" -> "/abs/path/IMG_0001.jpg"
    image_size_cache: Dict[str, Tuple[int, int]] = field(default_factory=dict)

    # ---------- History ----------
    history: List[Dict[str, Any]] = field(default_factory=list)
    hidx: int = -1

    # ---------- Save memo (view.save에서 사용) ----------
    _last_save_path: Optional[str] = None
    _last_save_fmt: Optional[str] = None

    # ---------- Convenience ----------
    def current_path(self) -> Optional[str]:
        if 0 <= self.current_index < len(self.image_files):
            return self.image_files[self.current_index]
        return None

    def boxes_for_current(self) -> List[Box]:
        p = self.current_path()
        if not p:
            return []
        return self.annotations.setdefault(p, [])

    def set_image_size(self, w: int, h: int) -> None:
        self.img_w, self.img_h = int(w), int(h)

    def snapshot(self) -> Dict[str, Any]:
        """Shallow-friendly snapshot; boxes deep-copied into lists."""
        ann_copy: Annotations = {}
        for k, v in self.annotations.items():
            ann_copy[k] = [Box(b.x1, b.y1, b.x2, b.y2, b.label, b.score) for b in v]
        return {
            "annotations": ann_copy,
            "class_list": list(self.class_list),
            "class_colors": dict(self.class_colors),
            "current_class": str(self.current_class),
        }

    def restore(self, snap: Dict[str, Any]) -> None:
        ann_in = snap.get("annotations", {})
        self.annotations = {
            k: [
                b if isinstance(b, Box)
                else Box.from_iter(b.to_list() if hasattr(b, "to_list") else b)
                for b in lst
            ]
            for k, lst in ann_in.items()
        }
        self.class_list = list(snap.get("class_list", ["object"]))
        self.class_colors = dict(snap.get("class_colors", {"object": "#00FF00"}))
        self.current_class = snap.get("current_class", "object")

    # ---------- Mutation helpers ----------
    def add_box_current(self, box: Box) -> None:
        self.boxes_for_current().append(box)

    def remove_indices_current(self, remove: Iterable[int]) -> None:
        idxs = sorted(set(int(i) for i in remove))
        boxes = self.boxes_for_current()
        kept = [b for i, b in enumerate(boxes) if i not in idxs]
        p = self.current_path()
        if p:
            self.annotations[p] = kept

    def remap_selection_after_delete(self, removed: Iterable[int]) -> None:
        removed = set(removed)
        new_sel: List[int] = []
        for i in self.selected_indices:
            if i in removed:
                continue
            dec = sum(1 for r in removed if r < i)
            nj = i - dec
            if nj >= 0:
                new_sel.append(nj)
        self.selected_indices = new_sel
