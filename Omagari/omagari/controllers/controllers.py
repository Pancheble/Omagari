from __future__ import annotations

from typing import Protocol, List, Iterable, Optional, Tuple, Dict
from tkinter import filedialog as fd, messagebox
import logging
import os

from ..domain import AppState, Box, normalize_box, clamp
from ..config import (
    HISTORY_MAX,
    NAV_INTERVAL_MS, NAV_HOLD_MS, SLIDER_DEBOUNCE_MS,
    YOLO_DEFAULT_CONF,
    DEFAULT_CLASS_NAME, DEFAULT_CLASS_COLOR,
    STATUS_READY_TEXT,
)
from ..services.yolo_service import YoloService



def _push_snapshot(state):
    snap = state.snapshot()
    state.history = state.history[: state.hidx + 1]
    state.history.append(snap)
    state.hidx += 1
    if len(state.history) > 100:
        state.history.pop(0)
        state.hidx -= 1


def undo(state, view):
    if state.hidx <= 0:
        messagebox.showinfo("Info", "No more undo", parent=view)
        return
    state.hidx -= 1
    snap = state.history[state.hidx]
    state.restore(snap)
    view.redraw()


def redo(state, view):
    if state.hidx + 1 >= len(state.history):
        messagebox.showinfo("Info", "No more redo", parent=view)
        return
    state.hidx += 1
    snap = state.history[state.hidx]
    state.restore(snap)
    view.redraw()

class ViewPort(Protocol):
    def request_render(self) -> None: ...
    def set_status(self, text: str) -> None: ...
    def update_title_index(self, now: int, total: int, basename: str) -> None: ...
    def load_image_to_view(self, path: str) -> Tuple[int, int]: ...
    def open_label_picker(self, box_index: int, x_root: Optional[int] = None, y_root: Optional[int] = None) -> None: ...
    def close_label_picker_if_any(self) -> None: ...
    def get_current_cv_image(self): ...
    def schedule_after_ms(self, ms: int, callback) -> None: ...


class NavController:
    def __init__(self, state):
        self.state = state

    def set_files(self, files):
        self.state.image_files = list(files)
        self.state.current_index = 0

    def show_next(self, view):
        if not self.state.image_files:
            return
        self.state.current_index = (self.state.current_index + 1) % len(self.state.image_files)
        path = self.state.current_path()
        if path:
            view.load_image_to_view(path)
            view._update_title()
            view.redraw()

    def show_prev(self, view):
        if not self.state.image_files:
            return
        self.state.current_index = (self.state.current_index - 1) % len(self.state.image_files)
        path = self.state.current_path()
        if path:
            view.load_image_to_view(path)
            view._update_title()
            view.redraw()

    def jump_to_index_1based(self, idx1, view):
        if not self.state.image_files:
            return
        i = max(1, min(int(idx1), len(self.state.image_files))) - 1
        if i == self.state.current_index:
            return
        self.state.current_index = i
        path = self.state.current_path()
        if path:
            view.load_image_to_view(path)
            view._update_title()
            view.redraw()

    def jump_to_last_annotated(self, view):
        last = -1
        for i, p in enumerate(self.state.image_files):
            if self.state.annotations.get(p):
                if len(self.state.annotations[p]) > 0:
                    last = i
        if last >= 0:
            self.state.current_index = last
            path = self.state.current_path()
            if path:
                view.load_image_to_view(path)
                view._update_title()
                view.redraw()


class LabelController:
    def __init__(self, state: AppState):
        self.state = state
        self._queue: List[int] = []
        self._is_toggle_on: bool = False

    def set_toggle_label(self, on: bool, viewport: ViewPort) -> None:
        self._is_toggle_on = bool(on)
        self.state.is_seq_labeling = bool(on)
        if not on:
            self._queue.clear()
            viewport.close_label_picker_if_any()
            viewport.set_status("Toggle Label: OFF")
        else:
            viewport.set_status("Toggle Label: ON")

    def feed_from_yolo(self, added_indices_with_conf: List[Tuple[int, float]], viewport: ViewPort) -> None:
        if not added_indices_with_conf:
            return
        if not self._is_toggle_on:
            self._queue.clear()
            return
        ordered = [i for i, _ in sorted(added_indices_with_conf, key=lambda t: -t[1])]
        self._queue = ordered
        self.state.is_seq_labeling = True
        self._open_next(viewport)

    def on_box_clicked_during_toggle(self, local_index: int, viewport: ViewPort, x_root=None, y_root=None) -> None:
        if not self._is_toggle_on:
            return
        try:
            if local_index in self._queue:
                self._queue.remove(local_index)
        except ValueError:
            pass
        self._queue.insert(0, local_index)
        self.state.selected_indices = [local_index]
        viewport.request_render()
        viewport.open_label_picker(local_index, x_root, y_root)

    def on_label_committed(self, box_index: int, new_label: str, viewport: ViewPort) -> None:
        boxes = self.state.boxes_for_current()
        if not (0 <= box_index < len(boxes)):
            return
        if new_label not in self.state.class_list:
            self.state.class_list.append(new_label)
        if new_label not in self.state.class_colors:
            self.state.class_colors[new_label] = DEFAULT_CLASS_COLOR
        boxes[box_index].label = new_label
        self.state.current_class = new_label
        self.state.selected_indices = [box_index]
        viewport.request_render()
        if self._is_toggle_on:
            try:
                if box_index in self._queue:
                    self._queue.remove(box_index)
            except ValueError:
                pass
            self._open_next(viewport)
        else:
            self.state.is_seq_labeling = False
            self._queue.clear()
            viewport.set_status("Toggle Label: done")

    def delete_selected(self, viewport: ViewPort) -> None:
        sel = list(sorted(set(self.state.selected_indices)))
        if not sel:
            return
        _push_snapshot(self.state)
        self.state.remove_indices_current(sel)
        self.state.remap_selection_after_delete(sel)
        self.state.selected_indices = []
        viewport.request_render()

    def clear_all_current(self, viewport: ViewPort) -> None:
        p = self.state.current_path()
        if not p:
            return
        _push_snapshot(self.state)
        self.state.annotations[p] = []
        self.state.selected_indices = []
        viewport.request_render()

    def _open_next(self, viewport: ViewPort) -> None:
        boxes = self.state.boxes_for_current()
        while self._queue and (self._queue[0] < 0 or self._queue[0] >= len(boxes)):
            self._queue.pop(0)
        if not self._queue:
            self.state.is_seq_labeling = False
            viewport.set_status("Toggle Label: done")
            return
        idx = self._queue.pop(0)
        self.state.selected_indices = [idx]
        viewport.request_render()
        viewport.open_label_picker(idx)




class YoloController:
    def __init__(self, state, service, device_manager, label_controller):
        self.state = state
        self.service = service
        self.device_manager = device_manager
        self.label_controller = label_controller

    def add_model(self, view):
        self.service.add_model_path_via_dialog(view)

    def select_model(self, view):
        self.service.select_and_load_model(view)

    def open_settings(self, view):
        self.service.open_model_settings_dialog(view)

    def show_runtime(self, view):
        self.service.show_runtime_dialog(view)

    def show_current_model(self, view):
        self.service.show_current_model_dialog(view)

    def infer_current_image(self, view, auto=False):
        self.service.infer_async(self.state, view, self.label_controller, auto=auto)
