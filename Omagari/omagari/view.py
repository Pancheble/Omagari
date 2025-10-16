# omagari/view.py
from __future__ import annotations
import os
import copy
import time
from typing import Optional, Tuple, List

import tkinter as tk
import tkinter.ttk as ttk
from tkinter import filedialog as fd, simpledialog, messagebox, colorchooser

import cv2
from PIL import Image, ImageTk

from .domain import AppState, Box, clamp, normalize_box, random_color, resource_path
from .config import (
    INITIAL_WINDOW_GEOMETRY, RIGHT_PANEL_MIN_WIDTH,
    SUPPORTED_IMAGE_EXTS, SAVE_FILETYPES, LOAD_ANNOT_FILETYPES,
    APP_TITLE, STATUS_READY_TEXT,
    SELECTED_BOX_OUTLINE_WIDTH, UNSELECTED_BOX_OUTLINE_WIDTH, FLASH_HIGHLIGHT_WIDTH,
    HANDLE_RADIUS_PX, DEFAULT_CLASS_NAME,
)
from .io.annotation_io import save_all as io_save_all, load_annotations as io_load_ann

# Controllers are injected by app.py
# from .controllers.controllers import NavController, LabelController, YoloController


class View(tk.Tk):
    """Tk view + event wiring. Uses injected controllers to mutate state."""

    def __init__(self, state: AppState, nav_ctrl, label_ctrl, yolo_ctrl):
        super().__init__()
        self.state = state
        self.nav = nav_ctrl
        self.label = label_ctrl
        self.yolo = yolo_ctrl

        self.title(APP_TITLE)
        self.geometry(INITIAL_WINDOW_GEOMETRY)
        try:
            self.iconphoto(False, tk.PhotoImage(file=resource_path(r"src\icon.png")))
        except Exception:
            pass

        self.status_var = tk.StringVar(value=STATUS_READY_TEXT)
        tk.Label(self, textvariable=self.status_var, anchor="w").pack(fill="x", side="bottom")

        self._yolo_auto_var = tk.BooleanVar(value=bool(self.state.yolo_auto_predict))
        self._toggle_label_var = tk.BooleanVar(value=bool(self.state.is_seq_labeling))

        self._idx_slider_var = tk.IntVar(value=0)
        self._idx_slider_suppress = False
        self._idx_slider = None

        self._flash_deadlines = {}
        self._crosshair_h_img = None
        self._crosshair_v_img = None

        self.cv_img = None
        self.pil_img = None
        self.tk_img = None

        self._build_menubar()
        self._build_body()
        self._bind_keys()

        self.after(50, lambda: self._set_sash_by_right_width(RIGHT_PANEL_MIN_WIDTH))

    # ------------ ViewPort protocol ------------

    def request_render(self) -> None:
        self.redraw()

    def set_status(self, text: str) -> None:
        self.status_var.set(text)

    def update_title_index(self, now: int, total: int, basename: str) -> None:
        self.title(f"{APP_TITLE} [{basename}] [{now}/{total}]")
        self.idx_progress.config(maximum=max(1, total), value=max(1, now))
        self.lbl_index.config(text=f"{now}/{total}")
        self._sync_index_slider()

    def load_image_to_view(self, path: str) -> Tuple[int, int]:
        img = cv2.imread(path)
        if img is None:
            messagebox.showerror("Error", f"Cannot load: {path}")
            return (self.state.img_w, self.state.img_h)
        self.cv_img = img
        h, w = img.shape[:2]
        self.state.set_image_size(w, h)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.pil_img = Image.fromarray(rgb)

        if not self.state.view_initialized:
            cw = max(200, self.canvas.winfo_width() or 800)
            ch = max(200, self.canvas.winfo_height() or 600)
            sx = cw / w
            sy = ch / h
            fit = min(1.0, sx, sy)
            self.state.scale = fit if fit > 0 else 1.0
            self.state.offset_x = (cw - w * self.state.scale) / 2
            self.state.offset_y = (ch - h * self.state.scale) / 2
            self.state.view_initialized = True

        self._update_tk_img()
        p = path
        if p not in self.state.annotations:
            self.state.annotations[p] = []
        self._sync_index_slider()
        return (w, h)

    def open_label_picker(self, box_index: int, x_root: Optional[int] = None, y_root: Optional[int] = None) -> None:
        self._open_label_picker(box_index, x_root, y_root)

    def close_label_picker_if_any(self) -> None:
        try:
            if hasattr(self, "label_pick_win") and self.label_pick_win and self.label_pick_win.winfo_exists():
                self.label_pick_win.destroy()
        except Exception:
            pass

    def get_current_cv_image(self):
        return self.cv_img

    def schedule_after_ms(self, ms: int, callback) -> None:
        self.after(int(ms), callback)

    # ------------ UI construction ------------

    def _build_menubar(self):
        menubar = tk.Menu(self)
        self.config(menu=menubar)

        m_file = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=m_file)
        m_file.add_command(label="Open Folder", command=self.open_folder)
        m_file.add_command(label="Prev Image", command=lambda: self.nav.show_prev(self))
        m_file.add_command(label="Next Image", command=lambda: self.nav.show_next(self))
        m_file.add_separator()
        m_file.add_command(label="Save Annotations", command=self.save)
        m_file.add_command(label="Save As Annotations", command=self.save_as)
        m_file.add_command(label="Load Annotations", command=self.load_annotations)
        m_file.add_separator()
        m_file.add_command(label="Exit", command=self.on_quit)

        m_edit = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Edit", menu=m_edit)
        m_edit.add_command(label="Undo", command=lambda: undo(self.state, self))
        m_edit.add_command(label="Redo", command=lambda: redo(self.state, self))
        m_edit.add_separator()
        m_edit.add_command(label="Delete Selected", command=lambda: self.label.delete_selected(self))
        m_edit.add_command(label="Clear All (current image)", command=lambda: self.label.clear_all_current(self))
        m_edit.add_command(label="Add Label...", command=self.add_label_dialog)
        m_edit.add_command(label="Label Setting", command=self.open_label_setting)
        m_edit.add_checkbutton(
            label="Toggle Label",
            variable=self._toggle_label_var,
            command=self._on_toggle_label_changed
        )
        m_edit.add_separator()
        m_edit.add_command(label="Zoom In", command=lambda: self._zoom_factor(1.2))
        m_edit.add_command(label="Zoom Out", command=lambda: self._zoom_factor(1/1.2))

        m_yolo = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="YOLO", menu=m_yolo)
        m_yolo.add_checkbutton(
            label="YOLO Model Auto Predict",
            variable=self._yolo_auto_var,
            command=self._toggle_yolo_auto_predict
        )
        m_yolo.add_separator()
        m_yolo.add_command(label="Add YOLO Model...",      command=lambda: self.yolo.add_model(self))
        m_yolo.add_command(label="Load YOLO Model...",     command=lambda: self.yolo.select_model(self))
        m_yolo.add_command(label="YOLO Model Settings...", command=lambda: self.yolo.open_settings(self))
        m_yolo.add_command(label="YOLO Model Runtime...",  command=lambda: self.yolo.show_runtime(self))
        m_yolo.add_command(label="YOLO Model",             command=lambda: self.yolo.show_current_model(self))

        m_setting = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Setting", menu=m_setting)
        m_setting.add_command(label="Device...", command=self.open_device_setting)

        m_help = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=m_help)
        m_help.add_command(
            label="About",
            command=lambda: (lambda w: (
                w.title("About"),
                w.resizable(False, False),
                setattr(w, "_img", tk.PhotoImage(file=resource_path(r"src\icon.png"))),
                tk.Label(w, image=w._img).pack(side="top", padx=10, pady=(10, 0)),
                tk.Label(w, text="mygf").pack(side="left", padx=10, pady=3),
                w.transient(self),
                w.attributes("-topmost", True),
                w.lift(),
                w.grab_set(),
                self.wait_window(w)
            ))(tk.Toplevel(self))
        )

    def _build_body(self):
        self.paned = tk.PanedWindow(self, orient="horizontal")
        self.paned.pack(fill="both", expand=True)
        self.left = tk.Frame(self.paned)
        self.right = tk.Frame(self.paned, width=RIGHT_PANEL_MIN_WIDTH)
        self.paned.add(self.left)
        self.paned.add(self.right, minsize=RIGHT_PANEL_MIN_WIDTH)

        self.canvas = tk.Canvas(self.left, bg="gray")
        self.canvas.pack(fill="both", expand=True)

        io = tk.Frame(self.right); io.pack(fill="x", padx=8)
        tk.Button(io, text="Save Annotations", command=self.save).pack(fill="x", pady=3)
        tk.Button(io, text="Save As Annotations", command=self.save_as).pack(fill="x", pady=3)
        tk.Button(io, text="Load Annotations", command=self.load_annotations).pack(fill="x", pady=3)

        nav = tk.Frame(self.right); nav.pack(fill="x", padx=8, pady=8)
        tk.Button(nav, text="<< Prev", command=lambda: self.nav.show_prev(self)).pack(side="left", fill="x", expand=True)
        tk.Button(nav, text="Next >>", command=lambda: self.nav.show_next(self)).pack(side="left", fill="x", expand=True, padx=3)
        tk.Button(nav, text="Open Folder", command=self.open_folder).pack(side="left", fill="x", expand=True)

        self._idx_slider = tk.Scale(
            self.right,
            from_=1, to=1,
            orient="horizontal",
            variable=self._idx_slider_var,
            showvalue=False,
            length=180,
            command=self._on_slider_change
        )
        self._idx_slider.pack(fill="x", padx=8, pady=(2, 0))

        self.idx_progress = ttk.Progressbar(self.right, mode="determinate", maximum=1, value=0)
        self.idx_progress.pack(fill="x", padx=8, pady=(6, 7))
        self.lbl_index = tk.Label(self.right, text="No image")
        self.lbl_index.pack(side="top", anchor="w", padx=6, pady=(3, 10))

        # Canvas bindings
        self.canvas.bind("<Button-1>",        self.on_lbutton_down)
        self.canvas.bind("<B1-Motion>",       self.on_mouse_move)
        self.canvas.bind("<ButtonRelease-1>", self.on_lbutton_up)
        self.canvas.bind("<Button-3>",        self.on_rbutton)
        self.canvas.bind("<MouseWheel>",      self.on_mouse_wheel)
        self.canvas.bind("<Button-4>",        self.on_mouse_wheel)
        self.canvas.bind("<Button-5>",        self.on_mouse_wheel)
        self.canvas.bind("<Motion>",          self.on_mouse_move_status)
        self.canvas.bind("<ButtonPress-2>",   self.on_mid_press)
        self.canvas.bind("<B2-Motion>",       self.on_mid_move)
        self.canvas.bind("<ButtonRelease-2>", self.on_mid_release)
        self.canvas.bind("<Leave>",           self._clear_crosshair)

        self.current_label_var = tk.StringVar(value=self.state.current_class)

        self.label_win = None
        self._lm_listbox = None
        self._lm_preview = None
        self.label_pick_win = None

        # mouse state
        self.mouse_action = None
        self.potential_create = False
        self.create_threshold = 6
        self.ix = 0
        self.iy = 0
        self.down_view = (0, 0)
        self.corner_thr = 12
        self.side_thr = 8
        self.grab_margin = 10
        self.active_resize = (None, None)

    def _bind_keys(self):
        self.bind_all("<KeyPress-Right>", lambda e: (self.nav.show_next(self), "break")[1])
        self.bind_all("<KeyPress-Left>",  lambda e: (self.nav.show_prev(self), "break")[1])
        self.bind_all("<KeyPress-d>",     lambda e: (self.nav.show_next(self), "break")[1])
        self.bind_all("<KeyPress-a>",     lambda e: (self.nav.show_prev(self), "break")[1])
        self.bind_all("<space>",          self._on_space)
        self.bind_all("<Delete>",         lambda e: (self.label.delete_selected(self), "break")[1])

        # Undo/Redo
        for seq in ("<Control-z>", "<Control-Z>"):
            self.bind_all(seq, lambda e: (undo(self.state, self), "break")[1])
        for seq in ("<Control-y>", "<Control-Y>", "<Control-Shift-z>", "<Control-Shift-Z>"):
            self.bind_all(seq, lambda e: (redo(self.state, self), "break")[1])

        # label hotkeys 1..9,0
        for i in range(1, 10):
            self.bind_all(str(i),      lambda e, idx=i: (self._relabel_selected_by_hotkey(idx), "break")[1])
            self.bind_all(f"<KP_{i}>", lambda e, idx=i: (self._relabel_selected_by_hotkey(idx), "break")[1])
        self.bind_all("0",      lambda e: (self._relabel_selected_by_hotkey(0), "break")[1])
        self.bind_all("<KP_0>", lambda e: (self._relabel_selected_by_hotkey(0), "break")[1])

        self.bind_all("<Control-s>", lambda e: (self.save(), "break")[1])

    # ------------ Layout helpers ------------

    def _set_sash_by_right_width(self, right_w: int):
        total = self.paned.winfo_width()
        if total <= 1:
            self.after(50, lambda: self._set_sash_by_right_width(right_w))
            return
        self.paned.sash_place(0, total - right_w, 0)

    # ------------ Coordinate transforms ------------

    def world_to_view(self, x, y):
        return x * self.state.scale + self.state.offset_x, y * self.state.scale + self.state.offset_y

    def view_to_world(self, vx, vy):
        return (vx - self.state.offset_x) / self.state.scale, (vy - self.state.offset_y) / self.state.scale

    def _zoom_factor(self, factor):
        new_scale = clamp(self.state.scale * factor, 0.1, 8.0)
        if abs(new_scale - self.state.scale) < 1e-6:
            return
        cx = self.canvas.winfo_width() // 2
        cy = self.canvas.winfo_height() // 2
        wx, wy = self.view_to_world(cx, cy)
        self.state.scale = new_scale
        self._update_tk_img()
        self.state.offset_x = cx - wx * self.state.scale
        self.state.offset_y = cy - wy * self.state.scale
        self.redraw()

    # ------------ File ops ------------

    def open_folder(self):
        folder = fd.askdirectory(title="Open Folder")
        if not folder:
            return
        try:
            files = sorted(
                os.path.join(folder, f)
                for f in os.listdir(folder)
                if f.lower().endswith(SUPPORTED_IMAGE_EXTS)
            )
        except Exception:
            files = []
        if not files:
            messagebox.showinfo("Info", "No images found.")
            return
        self.state.image_dir = folder
        self.nav.set_files(files)
        path = self.state.current_path()
        if path:
            self.load_image_to_view(path)
            self._update_title()
            self.redraw()
            self._sync_index_slider()

    def save(self):
        if getattr(self.state, "_last_save_path", None) and getattr(self.state, "_last_save_fmt", None):
            try:
                io_save_all(self.state, self.state._last_save_fmt, self.state._last_save_path)
                messagebox.showinfo("Saved", f"Saved to {self.state._last_save_path}")
                return
            except Exception as e:
                messagebox.showerror("Save failed", str(e))
                return
        self.save_as()

    def save_as(self):
        if not self.state.image_files:
            messagebox.showinfo("Info", "No images loaded.")
            return
        fpath = fd.asksaveasfilename(
            title="Save ALL annotations",
            defaultextension=".csv",
            filetypes=SAVE_FILETYPES,
        )
        if not fpath:
            return
        ext = os.path.splitext(fpath)[1].lower()
        if ext == ".csv":
            fmt = "CSV"
        elif ext == ".tsv":
            fmt = "TSV"
        elif ext == ".json":
            fmt = "JSON"
        elif ext == ".xml":
            fmt = "XML"
        else:
            fpath = fpath + ".csv"
            fmt = "CSV"

        try:
            io_save_all(self.state, fmt, fpath)
            self.state._last_save_path = fpath
            self.state._last_save_fmt = fmt
            messagebox.showinfo("Saved", f"Saved to {fpath}")
        except Exception as e:
            messagebox.showerror("Save failed", str(e))

    def load_annotations(self):
        if not self.state.image_files:
            messagebox.showinfo("Info", "Open an image folder first.")
            return
        fpath = fd.askopenfilename(
            title="Load Annotations (one file)",
            filetypes=LOAD_ANNOT_FILETYPES,
        )
        if not fpath:
            return
        try:
            self.canvas.configure(state="disabled")
        except Exception:
            pass
        t0 = time.time()
        loaded = io_load_ann(self.state, fpath)
        try:
            self.canvas.configure(state="normal")
        except Exception:
            pass

        if loaded <= 0:
            try:
                loaded = self._fallback_load_annotations(fpath)
            except Exception as e:
                messagebox.showerror("Load failed", f"Fallback parser error:\n{e}")
                loaded = 0

        if loaded > 0:
            self.nav.jump_to_last_annotated(self)
            path = self.state.current_path()
            if path:
                self.state.annotations.setdefault(path, [])
            self.redraw()
            elapsed = time.time() - t0
            messagebox.showinfo("Loaded", f"Loaded {loaded} boxes.\nTime: {elapsed:.2f}s")
        else:
            messagebox.showinfo(
                "Loaded",
                "No matching annotations found.\n\nTips:\n"
                "- Make sure you opened the image folder first.\n"
                "- Filenames in the annotation file should match images (basename match is OK)."
            )


    # ------------ Slider/title ------------

    def _on_slider_change(self, val):
        if self._idx_slider_suppress:
            return
        try:
            idx1 = int(float(val))
        except Exception:
            return
        if not self.state.image_files:
            return
        self.nav.jump_to_index_1based(idx1, self)

    def _sync_index_slider(self):
        if not self._idx_slider:
            return
        total = max(1, len(self.state.image_files) or 1)
        self._idx_slider_suppress = True
        try:
            self._idx_slider.config(from_=1, to=total)
            self._idx_slider_var.set(self.state.current_index + 1 if self.state.image_files else 1)
            state = ("normal" if self.state.image_files else "disabled")
            self._idx_slider.config(state=state)
        finally:
            self._idx_slider_suppress = False

    def _update_title(self):
        if not self.state.image_files:
            self.title(APP_TITLE)
            self.lbl_index.config(text="No image")
            self.idx_progress.config(maximum=1, value=0)
            return
        now = self.state.current_index + 1
        total = len(self.state.image_files)
        base = os.path.basename(self.state.current_path() or "")
        self.update_title_index(now, total, base)

    # ------------ Space/Auto predict ------------

    def _on_space(self, event=None):
        if not self.state.image_files:
            return "break"
        self.nav.show_next(self)
        if self._yolo_auto_var.get():
            self.yolo.infer_current_image(self, auto=True)
        return "break"

    def _toggle_yolo_auto_predict(self):
        self.state.yolo_auto_predict = bool(self._yolo_auto_var.get())

    # ------------ Device dialog (lightweight) ------------

    def open_device_setting(self):
        w = tk.Toplevel(self)
        w.title("Device Settings")
        w.resizable(False, False)
        w.attributes("-topmost", True)
        w.lift()

        mode_var = tk.StringVar(value=self.state.yolo_device_mode)
        tk.Radiobutton(w, text="CPU", variable=mode_var, value="CPU").pack(anchor="w", padx=10, pady=(10,0))

        # detect logical max
        from .services.device_manager import DeviceManager
        logical, _physical = DeviceManager.detect_cpu_threads()
        cpu_row = tk.Frame(w); cpu_row.pack(fill="x", padx=22, pady=(0,6))
        tk.Label(cpu_row, text=f"CPU Threads (max {logical}):").pack(side="left")
        thr_var = tk.IntVar(value=int(min(self.state.yolo_cpu_threads, logical)))
        tk.Spinbox(cpu_row, from_=1, to=logical, textvariable=thr_var, width=6).pack(side="left", padx=6)

        avail = DeviceManager.available_gpu_count()
        tk.Radiobutton(
            w, text=f"GPU (available: {avail})", variable=mode_var, value="GPU",
            state=("normal" if avail > 0 else "disabled")
        ).pack(anchor="w", padx=10)

        gpu_row = tk.Frame(w); gpu_row.pack(fill="x", padx=22, pady=(0,10))
        tk.Label(gpu_row, text="Number of GPUs:").pack(side="left")
        gpu_var = tk.IntVar(value=min(max(1, int(self.state.yolo_gpu_count)), max(1, avail)) if avail > 0 else 0)
        tk.Spinbox(
            gpu_row, from_=(1 if avail > 0 else 0), to=(avail if avail > 0 else 0),
            textvariable=gpu_var, width=6, state=("normal" if avail > 0 else "disabled")
        ).pack(side="left", padx=6)

        def apply():
            self.state.yolo_device_mode = mode_var.get()
            want_thr = int(thr_var.get())
            self.state.yolo_cpu_threads = max(1, min(want_thr, logical))
            if self.state.yolo_device_mode == "GPU":
                if avail <= 0:
                    messagebox.showinfo("Device", "No available GPU. Switching to CPU.")
                    self.state.yolo_device_mode = "CPU"
                    self.state.yolo_gpu_count = 0
                else:
                    self.state.yolo_gpu_count = max(1, min(int(gpu_var.get()), avail))
            else:
                self.state.yolo_gpu_count = 0
            messagebox.showinfo("Device", f"Settings applied: {self.state.yolo_device_mode} / cpu_threads={self.state.yolo_cpu_threads} / gpus={self.state.yolo_gpu_count}")
            w.destroy()

        tk.Button(w, text="Apply", command=apply).pack(fill="x", padx=10, pady=(0,10))

    # ------------ Drawing ------------

    def _update_tk_img(self):
        if not self.pil_img:
            return
        w = max(1, int(self.pil_img.width * self.state.scale))
        h = max(1, int(self.pil_img.height * self.state.scale))
        resized = self.pil_img.resize((w, h), Image.LANCZOS)
        self.tk_img = ImageTk.PhotoImage(resized)

    def redraw(self):
        self.canvas.delete("all")
        if self.tk_img:
            self.canvas.create_image(self.state.offset_x, self.state.offset_y, anchor="nw", image=self.tk_img)

        boxes = self.state.boxes_for_current()
        sel = set(self.state.selected_indices)
        for i, b in enumerate(boxes):
            x1, y1, x2, y2 = b.as_xyxy()
            vx1, vy1 = self.world_to_view(x1, y1)
            vx2, vy2 = self.world_to_view(x2, y2)
            color = "#FF3333" if i in sel else self.state.class_colors.get(b.label, "#00FF00")
            base_w = SELECTED_BOX_OUTLINE_WIDTH if i in sel else UNSELECTED_BOX_OUTLINE_WIDTH
            if i in self._flash_deadlines:
                base_w = max(base_w, FLASH_HIGHLIGHT_WIDTH)
            width = base_w
            self.canvas.create_rectangle(vx1, vy1, vx2, vy2, outline=color, width=width)
            text = b.label if b.score is None else f"{b.label} {b.score:.2f}"
            self.canvas.create_text(vx1 + 4, max(vy1 - 12, 4), text=text, anchor="nw", fill=color)
            if i in sel:
                for cx, cy in [(vx1, vy1), (vx2, vy1), (vx1, vy2), (vx2, vy2)]:
                    self.canvas.create_oval(cx - HANDLE_RADIUS_PX, cy - HANDLE_RADIUS_PX, cx + HANDLE_RADIUS_PX, cy + HANDLE_RADIUS_PX, fill=color, outline=color)

    # ------------ Hit testing ------------

    def _get_resize_type(self, wx, wy, box: Box):
        x1, y1, x2, y2 = box.as_xyxy()
        corners = [(x1, y1), (x2, y1), (x1, y2), (x2, y2)]
        for i, (cx, cy) in enumerate(corners):
            if abs(wx - cx) <= self.corner_thr and abs(wy - cy) <= self.corner_thr:
                return ('corner', i)
        if (y1 - self.side_thr) <= wy <= (y1 + self.side_thr) and x1 <= wx <= x2:
            return ('top', None)
        if (y2 - self.side_thr) <= wy <= (y2 + self.side_thr) and x1 <= wx <= x2:
            return ('bottom', None)
        if (x1 - self.side_thr) <= wx <= (x1 + self.side_thr) and y1 <= wy <= y2:
            return ('left', None)
        if (x2 - self.side_thr) <= wx <= (x2 + self.side_thr) and y1 <= wy <= y2:
            return ('right', None)
        return (None, None)

    def _is_near_border(self, wx, wy, box: Box, margin):
        x1, y1, x2, y2 = box.as_xyxy()
        left   = (x1 - margin) <= wx <= (x1 + margin) and y1 <= wy <= y2
        right  = (x2 - margin) <= wx <= (x2 + margin) and y1 <= wy <= y2
        top    = (y1 - margin) <= wy <= (y1 + margin) and x1 <= wx <= x2
        bottom = (y2 - margin) <= wy <= (y2 + margin) and x1 <= wx <= x2
        return left or right or top or bottom

    # ------------ Mouse events ------------

    def on_lbutton_down(self, event):
        vx, vy = event.x, event.y
        wx, wy = self.view_to_world(vx, vy)
        shift = (event.state & 0x0001) != 0
        ctrl  = (event.state & 0x0004) != 0
        self.ix, self.iy = vx, vy
        self.down_view = (vx, vy)
        self.potential_create = True
        self.mouse_action = None
        self.state.active_box = -1
        self.active_resize = (None, None)

        # Toggle Label short-circuit
        if self._toggle_label_var.get() and self.state.is_seq_labeling:
            boxes = self.state.boxes_for_current()
            for i in range(len(boxes) - 1, -1, -1):
                b = boxes[i]
                x1, y1, x2, y2 = b.as_xyxy()
                if x1 <= wx <= x2 and y1 <= wy <= y2:
                    self.state.selected_indices = [i]
                    self.redraw()
                    self.label.on_box_clicked_during_toggle(i, self, event.x_root, event.y_root)
                    return
            return

        boxes = self.state.boxes_for_current()
        for i in range(len(boxes) - 1, -1, -1):
            box = boxes[i]
            rtype, ridx = self._get_resize_type(wx, wy, box)
            if rtype is not None and shift:
                self.state.active_box = i
                self.mouse_action = 'resize'
                self.active_resize = (rtype, ridx)
                if not ctrl:
                    self.state.selected_indices = [i]
                else:
                    self.state.selected_indices = sorted(set(self.state.selected_indices + [i]))
                _push_snapshot(self.state)
                self.redraw()
                return
            if self._is_near_border(wx, wy, box, self.grab_margin) and not shift:
                self.state.active_box = i
                self.mouse_action = 'move'
                if not ctrl:
                    self.state.selected_indices = [i]
                else:
                    self.state.selected_indices = sorted(set(self.state.selected_indices + [i]))
                _push_snapshot(self.state)
                self.redraw()
                return
            x1, y1, x2, y2 = box.as_xyxy()
            if x1 <= wx <= x2 and y1 <= wy <= y2:
                self.state.active_box = i
                self.mouse_action = 'move'
                if not ctrl:
                    self.state.selected_indices = [i]
                else:
                    self.state.selected_indices = sorted(set(self.state.selected_indices + [i]))
                _push_snapshot(self.state)
                self.redraw()
                return

        # Ctrl drag => selection rect
        if ctrl:
            self.mouse_action = 'select_rect'
            self.sel_rect_start = (vx, vy)
            self.sel_rect_view = (vx, vy, vx, vy)
            return

        # hit none => clear selection
        self.state.selected_indices = []
        self.redraw()

    def on_mouse_move(self, event):
        vx, vy = event.x, event.y
        wx, wy = self.view_to_world(vx, vy)
        dx_view = vx - self.ix
        dy_view = vy - self.iy
        self.ix, self.iy = vx, vy

        boxes = self.state.boxes_for_current()

        if self.potential_create and self.mouse_action is None:
            sx, sy = self.down_view
            if abs(vx - sx) > self.create_threshold or abs(vy - sy) > self.create_threshold:
                self.mouse_action = 'create'
                self.potential_create = False
                wx0, wy0 = self.view_to_world(sx, sy)
                lbl = self.state.current_class or DEFAULT_CLASS_NAME
                boxes.append(Box(wx0, wy0, wx, wy, lbl, None))
                self.state.active_box = len(boxes) - 1
                self.state.selected_indices = [self.state.active_box]
                _push_snapshot(self.state)
                self.redraw()
                return

        if self.mouse_action == 'move':
            dx_world = dx_view / self.state.scale
            dy_world = dy_view / self.state.scale
            targets = self.state.selected_indices or ([self.state.active_box] if self.state.active_box >= 0 else [])
            for i in list(targets):
                b = boxes[i]
                moved = b.moved(dx_world, dy_world).clamp_to(self.state.img_w, self.state.img_h)
                boxes[i] = moved

        elif self.mouse_action == 'resize':
            i = self.state.active_box
            if i >= 0:
                rtype, ridx = self.active_resize
                dx_world = dx_view / self.state.scale
                dy_world = dy_view / self.state.scale
                b = boxes[i]
                x1, y1, x2, y2 = b.as_xyxy()
                if rtype == 'corner':
                    if ridx == 0:
                        x1 += dx_world; y1 += dy_world
                    elif ridx == 1:
                        x2 += dx_world; y1 += dy_world
                    elif ridx == 2:
                        x1 += dx_world; y2 += dy_world
                    elif ridx == 3:
                        x2 += dx_world; y2 += dy_world
                elif rtype == 'top':
                    y1 += dy_world
                elif rtype == 'bottom':
                    y2 += dy_world
                elif rtype == 'left':
                    x1 += dx_world
                elif rtype == 'right':
                    x2 += dx_world
                x1 = clamp(x1, 0, self.state.img_w - 1); x2 = clamp(x2, 0, self.state.img_w - 1)
                y1 = clamp(y1, 0, self.state.img_h - 1); y2 = clamp(y2, 0, self.state.img_h - 1)
                boxes[i] = Box(x1, y1, x2, y2, b.label, b.score)

        elif self.mouse_action == 'create':
            i = self.state.active_box
            if i >= 0:
                b = boxes[i]
                x0, y0, _, _ = b.x1, b.y1, b.x2, b.y2
                boxes[i] = Box(x0, y0, wx, wy, b.label, b.score)

        elif self.mouse_action == 'select_rect':
            x0, y0 = self.sel_rect_start
            self.sel_rect_view = (x0, y0, vx, vy)
            self.redraw()
            xA, yA, xB, yB = min(x0, vx), min(y0, vy), max(x0, vx), max(y0, vy)
            self.canvas.create_rectangle(xA, yA, xB, yB, outline="#33A", dash=(4, 2))
            return

        self.redraw()

    def on_lbutton_up(self, event):
        if self.mouse_action == 'create':
            boxes = self.state.boxes_for_current()
            for i in range(len(boxes)):
                b = boxes[i]
                x1, y1, x2, y2 = normalize_box(b.x1, b.y1, b.x2, b.y2)
                boxes[i] = Box(x1, y1, x2, y2, b.label, b.score)
            idx = self.state.active_box if self.state.active_box >= 0 else len(boxes) - 1
            if self._toggle_label_var.get():
                self._open_label_picker(idx, event.x_root, event.y_root)

        elif self.mouse_action == 'select_rect':
            x0, y0, x1, y1 = self.sel_rect_view
            vx0, vy0, vx1, vy1 = min(x0, x1), min(y0, y1), max(x0, y1 if False else x1), max(y0, y1)
            wx0, wy0 = self.view_to_world(vx0, vy0)
            wx1, wy1 = self.view_to_world(vx1, vy1)
            sel = []
            for i, b in enumerate(self.state.boxes_for_current()):
                bx1, by1, bx2, by2 = b.as_xyxy()
                if not (bx2 < min(wx0, wx1) or bx1 > max(wx0, wx1) or by2 < min(wy0, wy1) or by1 > max(wy0, wy1)):
                    sel.append(i)
            self.state.selected_indices = sel
            self.sel_rect_view = None

        self.potential_create = False
        self.mouse_action = None
        self.state.active_box = -1
        self.active_resize = (None, None)
        self.redraw()
        try:
            self.canvas.focus_set()
        except Exception:
            pass

    def on_rbutton(self, event):
        if getattr(event, "num", 3) != 3:
            return
        vx, vy = event.x, event.y
        wx, wy = self.view_to_world(vx, vy)
        boxes = self.state.boxes_for_current()
        for i in range(len(boxes) - 1, -1, -1):
            x1, y1, x2, y2 = boxes[i].as_xyxy()
            if x1 <= wx <= x2 and y1 <= wy <= y2:
                _push_snapshot(self.state)
                del boxes[i]
                # remap selection
                removed = {i}
                new_sel = []
                for j in self.state.selected_indices:
                    if j in removed: continue
                    dec = sum(1 for r in removed if r < j)
                    nj = j - dec
                    if nj >= 0: new_sel.append(nj)
                self.state.selected_indices = new_sel
                self.redraw()
                return

    def on_mouse_wheel(self, event):
        if hasattr(event, 'delta') and event.delta:
            delta = event.delta / 120.0
        else:
            if getattr(event, 'num', None) == 4:
                delta = 1
            elif getattr(event, 'num', None) == 5:
                delta = -1
            else:
                delta = 0
        factor = 1.15 ** delta
        self._zoom_factor(factor)

    def on_mid_press(self, event):
        self.pan_start = (event.x, event.y, self.state.offset_x, self.state.offset_y)

    def on_mid_move(self, event):
        sx, sy, ox0, oy0 = self.pan_start
        dx, dy = event.x - sx, event.y - sy
        self.state.offset_x = ox0 + dx
        self.state.offset_y = oy0 + dy
        self.redraw()

    def on_mid_release(self, event):
        if hasattr(self, "pan_start"):
            del self.pan_start

    # ------------ Crosshair (kept identical) ------------

    def on_mouse_move_status(self, event):
        vx, vy = event.x, event.y
        wx, wy = self.view_to_world(vx, vy)
        wxi, wyi = int(round(wx)), int(round(wy))

        crosshair_color = "#00FFFF"
        pix = "N/A"
        if 0 <= wxi < self.state.img_w and 0 <= wyi < self.state.img_h and self.cv_img is not None:
            b, g, r = self.cv_img[wyi, wxi]
            pix = f"RGB=({r},{g},{b})"
            ir, ig, ib = 255 - int(r), 255 - int(g), 255 - int(b)
            crosshair_color = f"#{ir:02x}{ig:02x}{ib:02x}"

        self.status_var.set(f"View=({vx},{vy}) World=({wx:.1f},{wy:.1f}) Pixel=({wxi},{wyi}) {pix}")
        self._draw_invert_crosshair(vx, vy)

    def _clear_crosshair(self, event=None):
        self.canvas.delete("crosshair")
        self._crosshair_h_img = None
        self._crosshair_v_img = None

    def _draw_invert_crosshair(self, vx, vy):
        self.canvas.delete("crosshair")

        if self.cv_img is None or self.tk_img is None:
            return

        w = self.canvas.winfo_width()
        h = self.canvas.winfo_height()

        bg_name = self.canvas['bg']
        r16, g16, b16 = self.winfo_rgb(bg_name)
        br, bgc, bb = (r16 // 256, g16 // 256, b16 // 256)
        inv_bg = (255 - br, 255 - bgc, 255 - bb, 255)

        wx_fixed = int(round((vx - self.state.offset_x) / self.state.scale))
        wy_fixed = int(round((vy - self.state.offset_y) / self.state.scale))

        from PIL import Image, ImageTk
        v_pixels = []
        for y in range(h):
            wy = int(round((y - self.state.offset_y) / self.state.scale))
            if (0 <= wx_fixed < self.state.img_w) and (0 <= wy < self.state.img_h):
                b, g, r = self.cv_img[wy, wx_fixed]
                v_pixels.append((255 - r, 255 - g, 255 - b, 255))
            else:
                v_pixels.append(inv_bg)
        v_img = Image.new("RGBA", (1, h))
        v_img.putdata(v_pixels)
        self._crosshair_v_img = ImageTk.PhotoImage(v_img)
        self.canvas.create_image(vx, 0, image=self._crosshair_v_img, anchor="nw", tags="crosshair")

        h_pixels = []
        for x in range(w):
            wx = int(round((x - self.state.offset_x) / self.state.scale))
            if (0 <= wx < self.state.img_w) and (0 <= wy_fixed < self.state.img_h):
                b, g, r = self.cv_img[wy_fixed, wx]
                h_pixels.append((255 - r, 255 - g, 255 - b, 255))
            else:
                h_pixels.append(inv_bg)
        h_img = Image.new("RGBA", (w, 1))
        h_img.putdata(h_pixels)
        self._crosshair_h_img = ImageTk.PhotoImage(h_img)
        self.canvas.create_image(0, vy, image=self._crosshair_h_img, anchor="nw", tags="crosshair")

    # ------------ Labeling UI ------------

    def _on_toggle_label_changed(self):
        on = bool(self._toggle_label_var.get())
        self.label.set_toggle_label(on, self)

    def add_label_dialog(self):
        new = simpledialog.askstring("Add Label", "Label name:", parent=self)
        if not new:
            return
        name = new.strip()
        if not name:
            return
        if name in self.state.class_list:
            messagebox.showinfo("Label", f"'{name}' already exists.")
            return
        self.state.class_list.append(name)
        self.state.class_colors[name] = self.state.class_colors.get(name, random_color())
        self.state.current_class = name
        self.current_label_var.set(name)
        self.status_var.set(f"Added label '{name}' and set as current.")
        try:
            self.canvas.focus_set()
        except Exception:
            pass

    def open_label_setting(self):
        if self.label_win and tk.Toplevel.winfo_exists(self.label_win):
            self.label_win.lift()
            return

        w = tk.Toplevel(self)
        w.title("Label Setting")
        w.geometry("420x420")
        w.resizable(False, False)
        w.attributes("-topmost", True)
        w.lift()
        self.label_win = w

        left = tk.Frame(w); left.pack(side="left", fill="both", expand=True, padx=8, pady=8)
        lb = tk.Listbox(left, exportselection=False); lb.pack(side="left", fill="both", expand=True)
        sb = tk.Scrollbar(left, orient="vertical", command=lb.yview); sb.pack(side="left", fill="y")
        lb.config(yscrollcommand=sb.set)
        self._lm_listbox = lb

        right = tk.Frame(w); right.pack(side="right", fill="y", padx=8, pady=8)
        self._lm_preview = tk.Canvas(right, width=140, height=40, highlightthickness=1, highlightbackground="#aaa")
        self._lm_preview.pack(pady=(0, 8))
        tk.Label(right, text="Preview (color)").pack(pady=(0, 8))

        tk.Button(right, text="Set Current", command=self._lm_set_current).pack(fill="x", pady=2)
        tk.Button(right, text="Add...", command=self._lm_add_via_dialog).pack(fill="x", pady=2)
        tk.Button(right, text="Rename...", command=self._lm_rename).pack(fill="x", pady=2)
        tk.Button(right, text="Delete", command=self._lm_delete).pack(fill="x", pady=2)
        tk.Button(right, text="Set Color...", command=self._lm_set_color).pack(fill="x", pady=2)
        tk.Button(right, text="Close", command=w.destroy).pack(fill="x", pady=(12, 0))

        lb.bind("<<ListboxSelect>>", self._lm_on_select)
        lb.bind("<Double-Button-1>", lambda e: self._lm_set_current())

        self._lm_refresh_list()

        def on_close():
            self._lm_listbox = None
            self._lm_preview = None
            self.label_win = None
            try:
                self.canvas.focus_set()
            except Exception:
                pass
            w.destroy()

        w.protocol("WM_DELETE_WINDOW", on_close)

    def _lm_refresh_list(self):
        if not self._lm_listbox or not self._lm_listbox.winfo_exists():
            self._lm_listbox = None
            if not self._lm_preview or not self._lm_preview.winfo_exists():
                self._lm_preview = None
            return
        lb = self._lm_listbox
        lb.delete(0, tk.END)
        for i, name in enumerate(self.state.class_list, start=1):
            marker = " *" if name == self.state.current_class else ""
            color = self.state.class_colors.get(name, "#00FF00")
            keytxt = "0" if i == 10 else (str(i) if i <= 9 else str(i))
            lb.insert(tk.END, f"[{keytxt}] {name}    [{color}]{marker}")
        self._lm_update_preview()

    def _lm_selected_name(self):
        if not self._lm_listbox or not self._lm_listbox.winfo_exists():
            self._lm_listbox = None
            return None
        sel = self._lm_listbox.curselection()
        if not sel:
            return None
        raw = self._lm_listbox.get(sel[0])
        tail = raw.split("] ", 1)[1]
        name = tail.split("    [", 1)[0]
        return name

    def _lm_update_preview(self):
        if not self._lm_preview or not self._lm_preview.winfo_exists():
            self._lm_preview = None
            return
        self._lm_preview.delete("all")
        name = self._lm_selected_name() or self.state.current_class
        color = self.state.class_colors.get(name, "#00FF00")
        self._lm_preview.create_rectangle(5, 5, 135, 35, fill=color, outline=color)
        self._lm_preview.create_text(70, 20, text=name, fill="white")

    def _lm_on_select(self, _evt=None):
        self._lm_update_preview()

    def _lm_set_current(self):
        name = self._lm_selected_name()
        if not name:
            return
        self.state.current_class = name
        self.current_label_var.set(name)
        self.status_var.set(f"Current label set to '{name}'")
        self._lm_refresh_list()

    def _lm_add_via_dialog(self):
        self.add_label_dialog()
        self._lm_refresh_list()

    def _lm_rename(self):
        name = self._lm_selected_name()
        if not name:
            return
        new = simpledialog.askstring("Rename Label", f"New name for '{name}':", initialvalue=name, parent=self.label_win or self)
        if not new:
            return
        new = new.strip()
        if not new or new == name:
            return
        if new in self.state.class_list:
            messagebox.showinfo("Label", f"'{new}' already exists.")
            return
        _push_snapshot(self.state)

        idx = self.state.class_list.index(name)
        self.state.class_list[idx] = new

        color = self.state.class_colors.pop(name, random_color())
        self.state.class_colors[new] = color

        for path, blist in self.state.annotations.items():
            for b in blist:
                if b.label == name:
                    b.label = new

        if self.state.current_class == name:
            self.state.current_class = new
            self.current_label_var.set(new)

        self._lm_refresh_list()
        self.redraw()

    def _lm_delete(self):
        name = self._lm_selected_name()
        if not name or name == DEFAULT_CLASS_NAME:
            if name == DEFAULT_CLASS_NAME:
                messagebox.showinfo("Label", f"Default label '{DEFAULT_CLASS_NAME}' cannot be deleted.")
            return
        if not messagebox.askyesno("Delete Label", f"Delete label '{name}'?\n(Boxes with this label will be changed to '{DEFAULT_CLASS_NAME}')"):
            return
        _push_snapshot(self.state)
        try:
            self.state.class_list.remove(name)
        except ValueError:
            pass
        self.state.class_colors.pop(name, None)
        if DEFAULT_CLASS_NAME not in self.state.class_list:
            self.state.class_list.insert(0, DEFAULT_CLASS_NAME)
            self.state.class_colors[DEFAULT_CLASS_NAME] = self.state.class_colors.get(DEFAULT_CLASS_NAME, "#00FF00")
        for path, blist in self.state.annotations.items():
            for b in blist:
                if b.label == name:
                    b.label = DEFAULT_CLASS_NAME
        if self.state.current_class == name:
            self.state.current_class = DEFAULT_CLASS_NAME
            self.current_label_var.set(DEFAULT_CLASS_NAME)
        self._lm_refresh_list()
        self.redraw()

    def _lm_set_color(self):
        name = self._lm_selected_name()
        if not name:
            return
        initial = self.state.class_colors.get(name, "#00FF00")
        rgb, hexv = colorchooser.askcolor(color=initial, title=f"Choose color for '{name}'")
        if not hexv:
            return
        self.state.class_colors[name] = hexv
        self._lm_refresh_list()
        self.redraw()
        try:
            self.canvas.focus_set()
        except Exception:
            pass

    # ------------ Label picker (popup) ------------

    def _open_label_picker(self, box_idx, x_root=None, y_root=None):
        try:
            if self.label_pick_win and self.label_pick_win.winfo_exists():
                self.label_pick_win.destroy()
        except Exception:
            pass

        w = tk.Toplevel(self)
        w.title("Choose Label")
        w.transient(self)
        w.resizable(False, False)
        w.attributes("-topmost", True)
        w.lift()
        if x_root is not None and y_root is not None:
            w.geometry(f"+{x_root+10}+{y_root+10}")
        self.label_pick_win = w

        info = tk.Label(w, text="Choose label: digits (1..9,0) or click", anchor="w")
        info.pack(fill="x", padx=8, pady=(8, 4))

        lb = tk.Listbox(w, exportselection=False, height=min(12, max(5, len(self.state.class_list))))
        lb.pack(fill="both", expand=True, padx=8, pady=(0, 8))

        for i, name in enumerate(self.state.class_list, start=1):
            keytxt = "0" if i == 10 else (str(i) if i <= 9 else " ")
            color = self.state.class_colors.get(name, "#00FF00")
            lb.insert(tk.END, f"[{keytxt}] {name}    [{color}]")

        try:
            cur_name = self.state.boxes_for_current()[box_idx].label
            if cur_name in self.state.class_list:
                lb.selection_set(self.state.class_list.index(cur_name))
                lb.see(self.state.class_list.index(cur_name))
        except Exception:
            pass

        def choose_by_index(i):
            if 0 <= i < len(self.state.class_list):
                try:
                    w.grab_release()
                except Exception:
                    pass
                w.destroy()
                self.after(0, lambda: self.label.on_label_committed(box_idx, self.state.class_list[i], self))
                return "break"

        def choose_from_list(_evt=None):
            sel = lb.curselection()
            if sel:
                return choose_by_index(sel[0])

        def on_escape(_e=None):
            self._toggle_label_var.set(False)
            self.label.set_toggle_label(False, self)
            try:
                w.destroy()
            except Exception:
                pass
            return "break"

        for i in range(1, 10):
            w.bind(str(i), lambda e, idx=i-1: choose_by_index(idx) or "break")
            w.bind(f"<KeyPress-{i}>", lambda e, idx=i-1: choose_by_index(idx) or "break")
        w.bind("0",           lambda e: (choose_by_index(9) if len(self.state.class_list) >= 10 else None) or "break")
        w.bind("<KeyPress-0>",lambda e: (choose_by_index(9) if len(self.state.class_list) >= 10 else None) or "break")
        w.bind("<Escape>", on_escape)

        lb.bind("<ButtonRelease-1>", choose_from_list)
        lb.bind("<Double-Button-1>", choose_from_list)

        w.grab_set()
        lb.focus_set()

    def _relabel_selected_by_hotkey(self, digit: int):
        try:
            if self.label_pick_win and tk.Toplevel.winfo_exists(self.label_pick_win):
                return
        except Exception:
            pass
        sel = self.state.selected_indices or ([self.state.active_box] if self.state.active_box >= 0 else [])
        if not sel:
            return
        target_idx = 9 if digit == 0 else digit - 1
        if target_idx < 0 or target_idx >= len(self.state.class_list):
            return
        new_label = self.state.class_list[target_idx]
        boxes = self.state.boxes_for_current()
        _push_snapshot(self.state)
        changed = []
        for i in list(sorted(set(sel))):
            if 0 <= i < len(boxes):
                boxes[i].label = new_label
                changed.append(i)
        if not changed:
            return
        self.state.current_class = new_label
        self.current_label_var.set(new_label)
        self.redraw()
        if self._toggle_label_var.get() and self.state.is_seq_labeling:
            pass
        else:
            self._toggle_label_var.set(False)
            self.label.set_toggle_label(False, self)

    def _fallback_load_annotations(self, fpath: str) -> int:
        import json, csv, os
        from .domain import Box, normalize_box

        stem_to_path = {}
        base_to_path = {}
        for p in self.state.image_files:
            base = os.path.basename(p)
            stem, _ = os.path.splitext(base)
            stem_to_path[stem.lower()] = p
            base_to_path[base.lower()] = p

        def _coerce_float(x):
            try:
                return float(x)
            except Exception:
                return None

        def _add_box(img_path, x1, y1, x2, y2, label, score):
            if img_path is None:
                return 0
            x1, y1, x2, y2 = normalize_box(float(x1), float(y1), float(x2), float(y2))
            b = Box(x1, y1, x2, y2, str(label), (None if score is None else float(score)))
            self.state.annotations.setdefault(img_path, []).append(b)
            return 1

        ext = os.path.splitext(fpath)[1].lower()
        total = 0

        if ext in (".csv", ".tsv"):
            delim = "," if ext == ".csv" else "\t"
            with open(fpath, "r", newline="", encoding="utf-8-sig") as f:
                reader = csv.DictReader(f, delimiter=delim)
                hdrs = [h.lower().strip() for h in reader.fieldnames or []]

                path_keys = [k for k in hdrs if k in ("path", "file", "image", "img", "filename")]
                x1_keys = [k for k in hdrs if k in ("x1","xmin","left")]
                y1_keys = [k for k in hdrs if k in ("y1","ymin","top")]
                x2_keys = [k for k in hdrs if k in ("x2","xmax","right")]
                y2_keys = [k for k in hdrs if k in ("y2","ymax","bottom")]
                label_keys = [k for k in hdrs if k in ("label","class","name")]
                score_keys = [k for k in hdrs if k in ("score","conf","confidence")]

                if not (path_keys and x1_keys and y1_keys and x2_keys and y2_keys and label_keys):
                    return 0

                pk, x1k, y1k, x2k, y2k, lk = path_keys[0], x1_keys[0], y1_keys[0], x2_keys[0], y2_keys[0], label_keys[0]
                sk = score_keys[0] if score_keys else None

                for row in reader:
                    rawp = (row.get(pk) or "").strip()
                    if not rawp:
                        continue
                    base = os.path.basename(rawp).lower()
                    stem = os.path.splitext(base)[0].lower()
                    img_path = base_to_path.get(base) or stem_to_path.get(stem)

                    x1 = _coerce_float(row.get(x1k))
                    y1 = _coerce_float(row.get(y1k))
                    x2 = _coerce_float(row.get(x2k))
                    y2 = _coerce_float(row.get(y2k))
                    lab = row.get(lk) or "object"
                    sco = _coerce_float(row.get(sk)) if sk else None

                    if img_path and None not in (x1,y1,x2,y2):
                        total += _add_box(img_path, x1, y1, x2, y2, lab, sco)

        elif ext == ".json":
            with open(fpath, "r", encoding="utf-8") as f:
                data = json.load(f)
            items = []
            if isinstance(data, dict) and "images" in data:
                for it in data["images"]:
                    rawp = (it.get("file") or it.get("path") or it.get("image") or "")
                    base = os.path.basename(rawp).lower()
                    stem = os.path.splitext(base)[0].lower()
                    img_path = base_to_path.get(base) or stem_to_path.get(stem)
                    for bx in (it.get("boxes") or []):
                        x1 = _coerce_float(bx.get("x1") or bx.get("xmin") or bx.get("left"))
                        y1 = _coerce_float(bx.get("y1") or bx.get("ymin") or bx.get("top"))
                        x2 = _coerce_float(bx.get("x2") or bx.get("xmax") or bx.get("right"))
                        y2 = _coerce_float(bx.get("y2") or bx.get("ymax") or bx.get("bottom"))
                        lab = bx.get("label") or bx.get("class") or "object"
                        sco = _coerce_float(bx.get("score") or bx.get("conf") or bx.get("confidence"))
                        if img_path and None not in (x1,y1,x2,y2):
                            total += _add_box(img_path, x1, y1, x2, y2, lab, sco)
            else:
                if isinstance(data, dict) and "annotations" in data:
                    items = data["annotations"]
                elif isinstance(data, list):
                    items = data
                for it in items:
                    rawp = (it.get("path") or it.get("file") or it.get("image") or "")
                    base = os.path.basename(rawp).lower()
                    stem = os.path.splitext(base)[0].lower()
                    img_path = base_to_path.get(base) or stem_to_path.get(stem)
                    x1 = _coerce_float(it.get("x1") or it.get("xmin") or it.get("left"))
                    y1 = _coerce_float(it.get("y1") or it.get("ymin") or it.get("top"))
                    x2 = _coerce_float(it.get("x2") or it.get("xmax") or it.get("right"))
                    y2 = _coerce_float(it.get("y2") or it.get("ymax") or it.get("bottom"))
                    lab = it.get("label") or it.get("class") or "object"
                    sco = _coerce_float(it.get("score") or it.get("conf") or it.get("confidence"))
                    if img_path and None not in (x1,y1,x2,y2):
                        total += _add_box(img_path, x1, y1, x2, y2, lab, sco)

        return total


    # ------------ YOLO trigger ------------

    def run_yolo_on_current_image(self, auto=False):
        self.yolo.infer_current_image(self, auto=auto)

    # ------------ Quit ------------

    def on_quit(self):
        try:
            self.destroy()
        except Exception:
            pass


# imports placed here to avoid circulars in type-checkers
from .controllers.controllers import undo, redo, _push_snapshot
