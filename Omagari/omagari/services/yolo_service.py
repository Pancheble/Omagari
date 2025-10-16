from __future__ import annotations
import threading
import tkinter as tk
from tkinter import filedialog as fd, messagebox
from ultralytics import YOLO
from ..domain import Box, clamp

class YoloService:
    def __init__(self, state):
        self.state = state
        self.model = None
        self._busy = False

    def add_model_path_via_dialog(self, view):
        path = fd.askopenfilename(title="Add YOLO Model", filetypes=[("PyTorch Model", "*.pt *.pth"), ("All", "*.*")])
        if not path:
            return
        if path not in self.state.yolo_model_paths:
            self.state.yolo_model_paths.append(path)
        view.set_status(f"Model added: {path}")

    def select_and_load_model(self, view):
        paths = list(self.state.yolo_model_paths or [])
        if not paths:
            messagebox.showinfo("YOLO", "먼저 'Add YOLO Model...'로 모델을 추가하세요.")
            return
        w = tk.Toplevel(view)
        w.title("Load YOLO Model")
        w.resizable(False, False)
        w.attributes("-topmost", True)
        w.lift()
        lb = tk.Listbox(w, width=70, height=min(12, len(paths)))
        for p in paths:
            lb.insert(tk.END, p)
        lb.pack(side="left", fill="both", expand=True, padx=(8,0), pady=8)
        sb = tk.Scrollbar(w, orient="vertical", command=lb.yview)
        sb.pack(side="left", fill="y", padx=(0,8), pady=8)
        lb.config(yscrollcommand=sb.set)
        def do_load():
            sel = lb.curselection()
            if not sel:
                return
            path = paths[sel[0]]
            try:
                view.set_status("Loading YOLO model...")
                view.update_idletasks()
            except Exception:
                pass
            try:
                m = YOLO(path)
                self.model = m
                self.state.current_yolo_model_path = path
                names = getattr(m, "names", {})
                if isinstance(names, dict):
                    cls_names = [names[i] for i in sorted(names.keys())]
                else:
                    cls_names = list(names)
                if cls_names:
                    self.state.yolo_enabled_classes = set(cls_names)
                    for n in cls_names:
                        if n not in self.state.class_list:
                            self.state.class_list.append(n)
                messagebox.showinfo("YOLO", f"Model loaded:\n{path}")
                w.destroy()
            except Exception as e:
                messagebox.showerror("YOLO", f"Failed to load model:\n{e}")
        tk.Button(w, text="Load", command=do_load).pack(fill="x", padx=8, pady=(0,8))
        lb.bind("<Double-Button-1>", lambda _e: do_load())

    def open_model_settings_dialog(self, view):
        if self.model is None:
            messagebox.showinfo("YOLO", "Load YOLO Model 먼저 실행하세요.")
            return
        top = tk.Toplevel(view)
        top.title("YOLO Model Settings")
        top.resizable(False, False)
        top.attributes("-topmost", True)
        top.lift()
        frm = tk.Frame(top); frm.pack(fill="x", padx=10, pady=10)
        tk.Label(frm, text="Confidence").pack(anchor="w")
        conf_var = tk.DoubleVar(value=float(self.state.yolo_conf))
        tk.Scale(frm, variable=conf_var, orient="horizontal", from_=0.0, to=1.0, resolution=0.01, length=260).pack(fill="x")
        names = getattr(self.model, "names", {})
        if isinstance(names, dict):
            items = [(i, names[i]) for i in sorted(names.keys())]
        else:
            items = list(enumerate(names))
        cls_frame = tk.LabelFrame(top, text="Class On/Off"); cls_frame.pack(fill="both", expand=True, padx=10, pady=(0,10))
        canvas = tk.Canvas(cls_frame, height=220)
        scroll = tk.Scrollbar(cls_frame, orient="vertical", command=canvas.yview)
        inner = tk.Frame(canvas)
        inner.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0,0), window=inner, anchor="nw")
        canvas.configure(yscrollcommand=scroll.set)
        canvas.pack(side="left", fill="both", expand=True)
        scroll.pack(side="left", fill="y")
        enabled_now = set(self.state.yolo_enabled_classes) if self.state.yolo_enabled_classes else {n for _, n in items}
        cls_vars = {}
        for _cid, cname in items:
            v = tk.BooleanVar(value=(cname in enabled_now))
            tk.Checkbutton(inner, text=cname, variable=v).pack(anchor="w")
            cls_vars[cname] = v
        btns = tk.Frame(top); btns.pack(fill="x", padx=10, pady=(0,10))
        def select_all(v):
            for x in cls_vars.values():
                x.set(v)
        tk.Button(btns, text="Select All", command=lambda: select_all(True)).pack(side="left", padx=4)
        tk.Button(btns, text="Deselect All", command=lambda: select_all(False)).pack(side="left", padx=4)
        def apply_close():
            self.state.yolo_conf = float(conf_var.get())
            enabled = {name for name, var in cls_vars.items() if var.get()}
            self.state.yolo_enabled_classes = enabled if len(enabled) < len(cls_vars) else set(cls_vars.keys())
            top.destroy()
        tk.Button(btns, text="Save", command=apply_close).pack(side="right")

    def show_runtime_dialog(self, view):
        try:
            import torch, cv2
            th = getattr(torch, "get_num_threads", lambda: -1)()
            ith = getattr(torch, "get_num_interop_threads", lambda: -1)()
            cv_th = cv2.getNumThreads()
            cuda = torch.cuda.is_available()
            msg = (
                f"Device mode: {self.state.yolo_device_mode}\n"
                f"Device str: {self._device_string()}\n"
                f"torch threads: {th} (interop {ith})\n"
                f"cv2 threads: {cv_th}\n"
                f"CUDA available: {cuda}\n"
            )
        except Exception as e:
            msg = f"Runtime info unavailable: {e}"
        messagebox.showinfo("YOLO Runtime", msg)

    def show_current_model_dialog(self, view):
        dev = self._device_string()
        msg = ("No YOLO model currently in use."
               if not self.state.current_yolo_model_path
               else f"Current model:\n{self.state.current_yolo_model_path}\n\nConfidence: {self.state.yolo_conf}\nDevice: {dev}")
        messagebox.showinfo("YOLO", msg)

    def infer_async(self, state, view, label_controller=None, auto=False):
        if self.model is None:
            messagebox.showinfo("YOLO", "Please load a YOLO model first.")
            return
        if self._busy:
            return
        img = view.get_current_cv_image()
        if img is None:
            messagebox.showinfo("YOLO", "Open an image first.")
            return
        self._busy = True
        def job():
            try:
                dev = self._device_string()
                try:
                    self.model.to("cpu" if dev == "cpu" else "cuda:0")
                except Exception:
                    pass
                results = self.model(img, conf=float(state.yolo_conf), device=dev, verbose=False)
            except Exception as e:
                self._busy = False
                messagebox.showerror("YOLO", f"Inference failed: {e}")
                return
            added = []
            try:
                res = results[0]
                if not hasattr(res, "boxes") or res.boxes is None or res.boxes.xyxy is None:
                    raise RuntimeError("No boxes")
                names = getattr(self.model, "names", None) or getattr(res, "names", {})
                def name_of(cid):
                    if isinstance(names, dict):
                        return names.get(int(cid), f"cls{int(cid)}")
                    idx = int(cid)
                    return names[idx] if 0 <= idx < len(names) else f"cls{idx}"
                xyxys = res.boxes.xyxy.cpu().numpy().tolist() if hasattr(res.boxes.xyxy, "cpu") else res.boxes.xyxy.tolist()
                clss = res.boxes.cls
                confs = res.boxes.conf
                path = state.current_path()
                if not path:
                    self._busy = False
                    return
                state.annotations.setdefault(path, [])
                w, h = state.img_w, state.img_h
                for i, xyxy in enumerate(xyxys):
                    cid = int(clss[i].item()) if clss is not None else 0
                    conf = float(confs[i].item()) if confs is not None else None
                    label = str(name_of(cid))
                    if state.yolo_enabled_classes is not None and label not in state.yolo_enabled_classes:
                        continue
                    x1, y1, x2, y2 = xyxy
                    x1 = clamp(x1, 0, w - 1); x2 = clamp(x2, 0, w - 1)
                    y1 = clamp(y1, 0, h - 1); y2 = clamp(y2, 0, h - 1)
                    if label not in state.class_list:
                        state.class_list.append(label)
                    state.annotations[path].append(Box(int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2)), label, conf))
                    added.append((len(state.annotations[path]) - 1, conf if conf is not None else 0.0))
            finally:
                self._busy = False
                def after_merge():
                    view.request_render()
                    if added and label_controller is not None:
                        label_controller.feed_from_yolo(added, view)
                    if added and (not state.is_seq_labeling):
                        try:
                            view.open_label_picker(added[0][0])
                        except Exception:
                            pass
                try:
                    view.schedule_after_ms(0, after_merge)
                except Exception:
                    after_merge()
        threading.Thread(target=job, daemon=True).start()

    def _device_string(self):
        try:
            import torch
            if self.state.yolo_device_mode == "GPU" and torch.cuda.is_available():
                n = max(1, int(self.state.yolo_gpu_count))
                n = min(n, torch.cuda.device_count() if torch.cuda.is_available() else 1)
                return ",".join(str(i) for i in range(n))
        except Exception:
            pass
        return "cpu"

    def shutdown(self):
        pass
