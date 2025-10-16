# omagari/io/annotation_io.py
"""
Robust annotation I/O (CSV/TSV/JSON/XML/YOLO TXT) using only standard libs.

- No UI or dialogs here.
- Strict-but-forgiving parsing: silently skip malformed rows, never crash.
- Works against AppState (domain) and constants (config).
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import os
import csv
import json
import xml.etree.ElementTree as ET

from ..domain import AppState, Box, normalize_box, is_number, resource_path, random_color
from ..config import (
    SAVE_FILETYPES, LOAD_ANNOT_FILETYPES,
    CSV_TSV_STRICT, XML_STRICT, JSON_STRICT, YOLO_TXT_STRICT,
    DEFAULT_CLASS_NAME, DEFAULT_CLASS_COLOR
)

# --------------------------- Public API ---------------------------

def save_all(state: AppState, fmt: str, path: str) -> None:
    """Serialize all annotations from state into the given file format."""
    fmt = (fmt or "").upper()
    if fmt in ("CSV", "TSV") or path.lower().endswith(".csv") or path.lower().endswith(".tsv"):
        _save_csv_tsv(state, path)
    elif fmt == "JSON" or path.lower().endswith(".json"):
        _save_json(state, path)
    elif fmt == "XML" or path.lower().endswith(".xml"):
        _save_xml(state, path)
    else:
        # default to CSV
        if not path.lower().endswith(".csv"):
            path = path + ".csv"
        _save_csv_tsv(state, path)


def load_annotations(state: AppState, path: str) -> int:
    """
    Load annotations from a single file into state.annotations.
    Returns count of boxes loaded (matching files only).
    """
    ext = os.path.splitext(path)[1].lower()
    if ext in (".csv", ".tsv"):
        mapping = _parse_csv_or_tsv(state, path)
    elif ext == ".json":
        mapping = _parse_json_file(state, path)
    elif ext == ".xml":
        mapping = _parse_xml_file(state, path)
    elif ext == ".txt":
        mapping = _parse_txt_yolo_aggregated_or_single(state, path)
    else:
        return 0
    return _merge_mapping_into_state(state, mapping)

# --------------------------- Saving ---------------------------

def _save_csv_tsv(state: AppState, path: str) -> None:
    delim = "\t" if path.lower().endswith(".tsv") else ","
    header = ["dir", "label", "x1", "y1", "x2", "y2"]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter=delim)
        writer.writerow(header)
        for img_path in state.image_files:
            imgname = os.path.basename(img_path)
            for b in state.annotations.get(img_path, []):
                x1, y1, x2, y2 = b.as_xyxy()
                writer.writerow([
                    imgname, str(b.label),
                    int(round(x1)), int(round(y1)),
                    int(round(x2)), int(round(y2)),
                ])

def _save_json(state: AppState, path: str) -> None:
    payload = {"images": []}
    for img_path in state.image_files:
        imgname = os.path.basename(img_path)
        boxes = []
        for b in state.annotations.get(img_path, []):
            x1, y1, x2, y2 = b.as_xyxy()
            boxes.append([
                int(round(x1)), int(round(y1)),
                int(round(x2)), int(round(y2)),
                b.label, b.score
            ])
        payload["images"].append({"file": imgname, "boxes": boxes})
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

def _save_xml(state: AppState, path: str) -> None:
    root = ET.Element("annotations")
    for img_path in state.image_files:
        imgname = os.path.basename(img_path)
        e_img = ET.SubElement(root, "image", {"file": imgname})
        for b in state.annotations.get(img_path, []):
            x1, y1, x2, y2 = b.as_xyxy()
            e_obj = ET.SubElement(e_img, "object")
            ET.SubElement(e_obj, "name").text = str(b.label)
            e_box = ET.SubElement(e_obj, "bndbox")
            ET.SubElement(e_box, "xmin").text = str(int(round(x1)))
            ET.SubElement(e_box, "ymin").text = str(int(round(y1)))
            ET.SubElement(e_box, "xmax").text = str(int(round(x2)))
            ET.SubElement(e_box, "ymax").text = str(int(round(y2)))
    tree = ET.ElementTree(root)
    # Pretty print (Py3.9+)
    try:
        ET.indent(tree, space="  ", level=0)
    except Exception:
        pass
    with open(path, "w", encoding="utf-8") as f:
        tree.write(f, encoding="unicode")

# --------------------------- Loading / Parsing ---------------------------

def _parse_csv_or_tsv(state: AppState, path: str) -> Dict[str, List[Box]]:
    """
    Accepts multiple header conventions:
      - dir,label,x1,y1,x2,y2
      - image_filename + coordinates (+ optional label)
    Also supports headerless numeric rows heuristically.
    """
    mapping: Dict[str, List[Box]] = {}
    with open(path, "r", encoding="utf-8", newline="") as f:
        sample = f.read(2048)
        f.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample)
        except Exception:
            dialect = csv.excel
        reader = csv.reader(f, dialect)
        rows = [r for r in reader if any(c.strip() for c in r)]

    if not rows:
        return mapping

    header = [c.strip().lower() for c in rows[0]]
    data_rows = rows[1:] if any(not is_number(c) for c in header) else rows

    def add_row(imgname: str, x1: str, y1: str, x2: str, y2: str, label: Optional[str]) -> None:
        try:
            bx = Box(float(x1), float(y1), float(x2), float(y2), (label or DEFAULT_CLASS_NAME), None)
        except Exception:
            return
        base = os.path.splitext(os.path.basename(imgname))[0]
        mapping.setdefault(base, []).append(bx)

    # Headered variants
    if any(not is_number(c) for c in header):
        h = {k: i for i, k in enumerate(header)}
        # Format: dir,label,x1,y1,x2,y2
        if {"dir", "x1", "y1", "x2", "y2"} <= set(h.keys()):
            for r in data_rows:
                try:
                    imgname = r[h["dir"]]
                    x1, y1, x2, y2 = r[h["x1"]], r[h["y1"]], r[h["x2"]], r[h["y2"]]
                    label = r[h["label"]] if "label" in h and h["label"] < len(r) else DEFAULT_CLASS_NAME
                    add_row(imgname, x1, y1, x2, y2, label)
                except Exception:
                    continue
        # Format: image_filename + coords (+ optional label)
        elif "image_filename" in h:
            for r in data_rows:
                try:
                    imgname = r[h["image_filename"]]
                    x1 = r[h["x1"]] if "x1" in h else r[h["xmin"]] if "xmin" in h else r[1]
                    y1 = r[h["y1"]] if "y1" in h else r[h["ymin"]] if "ymin" in h else r[2]
                    x2 = r[h["x2"]] if "x2" in h else r[h["xmax"]] if "xmax" in h else r[3]
                    y2 = r[h["y2"]] if "y2" in h else r[h["ymax"]] if "ymax" in h else r[4]
                    label = r[h["label"]] if "label" in h and h["label"] < len(r) else DEFAULT_CLASS_NAME
                    add_row(imgname, x1, y1, x2, y2, label)
                except Exception:
                    continue
        else:
            # Fallback: try to guess (img, label, x1, y1, x2, y2)
            for r in data_rows:
                if len(r) >= 6 and (not is_number(r[0])) and all(is_number(c) for c in r[2:6]):
                    add_row(r[0], r[2], r[3], r[4], r[5], r[1])
                elif len(r) >= 5 and all(is_number(c) for c in r[0:4]):
                    # No filename, assume .csv basename equals file base
                    base = os.path.splitext(os.path.basename(path))[0]
                    try:
                        bx = Box(float(r[0]), float(r[1]), float(r[2]), float(r[3]), r[4], None)
                        mapping.setdefault(base, []).append(bx)
                    except Exception:
                        continue
                elif len(r) >= 6 and (not is_number(r[0])) and all(is_number(c) for c in r[1:5]):
                    add_row(r[0], r[1], r[2], r[3], r[4], r[5])
    else:
        # Headerless numeric rows
        for r in data_rows:
            if len(r) >= 5 and all(is_number(c) for c in r[0:4]):
                base = os.path.splitext(os.path.basename(path))[0]
                try:
                    bx = Box(float(r[0]), float(r[1]), float(r[2]), float(r[3]), r[4], None)
                    mapping.setdefault(base, []).append(bx)
                except Exception:
                    continue

    return mapping


def _parse_json_file(state: AppState, path: str) -> Dict[str, List[Box]]:
    mapping: Dict[str, List[Box]] = {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
    except Exception:
        return mapping

    if isinstance(obj, dict) and "images" in obj:
        for rec in obj["images"]:
            base = os.path.splitext(os.path.basename(rec.get("file", "")))[0]
            lst: List[Box] = []
            for b in rec.get("boxes", []):
                if len(b) >= 5:
                    try:
                        x1, y1, x2, y2, label = float(b[0]), float(b[1]), float(b[2]), float(b[3]), str(b[4])
                        score = (None if len(b) < 6 or b[5] is None else float(b[5]))
                        lst.append(Box(x1, y1, x2, y2, label, score))
                    except Exception:
                        continue
            if lst:
                mapping[base] = lst
    elif isinstance(obj, dict) and "image" in obj:
        base = os.path.splitext(os.path.basename(obj.get("image", "")))[0]
        lst: List[Box] = []
        for b in obj.get("boxes", []):
            if len(b) >= 5:
                try:
                    x1, y1, x2, y2, label = float(b[0]), float(b[1]), float(b[2]), float(b[3]), str(b[4])
                    score = (None if len(b) < 6 or b[5] is None else float(b[5]))
                    lst.append(Box(x1, y1, x2, y2, label, score))
                except Exception:
                    continue
        if lst:
            mapping[base] = lst

    return mapping


def _parse_xml_file(state: AppState, path: str) -> Dict[str, List[Box]]:
    """
    Supports:
      1) Aggregated: <annotations><image file="..."><object>...</object></image>...</annotations>
      2) Pascal VOC: <annotation><filename>...</filename><object>...</object>...</annotation>
    """
    mapping: Dict[str, List[Box]] = {}
    try:
        text = open(path, "r", encoding="utf-8").read()
    except Exception:
        return mapping

    text_stripped = text.strip()
    if "<annotations" in text_stripped and "<image" in text_stripped:
        # Aggregated
        try:
            root = ET.fromstring(text_stripped)
        except Exception:
            return mapping
        for e_img in root.findall(".//image"):
            file_attr = e_img.attrib.get("file", "").strip()
            base = os.path.splitext(os.path.basename(file_attr))[0]
            lst: List[Box] = []
            for e_obj in e_img.findall(".//object"):
                name = (e_obj.findtext("name") or DEFAULT_CLASS_NAME).strip()
                try:
                    xmin = float(e_obj.findtext("bndbox/xmin"))
                    ymin = float(e_obj.findtext("bndbox/ymin"))
                    xmax = float(e_obj.findtext("bndbox/xmax"))
                    ymax = float(e_obj.findtext("bndbox/ymax"))
                    lst.append(Box(xmin, ymin, xmax, ymax, name, None))
                except Exception:
                    continue
            if lst:
                mapping.setdefault(base, []).extend(lst)
        return mapping

    # Pascal VOC
    try:
        root = ET.fromstring(text_stripped)
    except Exception:
        return mapping
    fname = (root.findtext("filename") or "").strip()
    if not fname:
        return mapping
    base = os.path.splitext(os.path.basename(fname))[0]
    lst: List[Box] = []
    for e_obj in root.findall(".//object"):
        name = (e_obj.findtext("name") or DEFAULT_CLASS_NAME).strip()
        try:
            xmin = float(e_obj.findtext("bndbox/xmin"))
            ymin = float(e_obj.findtext("bndbox/ymin"))
            xmax = float(e_obj.findtext("bndbox/xmax"))
            ymax = float(e_obj.findtext("bndbox/ymax"))
            lst.append(Box(xmin, ymin, xmax, ymax, name, None))
        except Exception:
            continue
    if lst:
        mapping[base] = lst
    return mapping


def _parse_txt_yolo_aggregated_or_single(state: AppState, path: str) -> Dict[str, List[Box]]:
    """
    Two modes:
      A) Aggregated text with sections: lines starting with '### IMAGE: filename'
      B) Single-image YOLO .txt (class cx cy w h) -> applied to current image basename
    """
    mapping: Dict[str, List[Box]] = {}
    with open(path, "r", encoding="utf-8") as f:
        lines = [l.rstrip("\n") for l in f]

    if any(l.startswith("### IMAGE:") for l in lines):
        cur_base: Optional[str] = None
        cur_path: Optional[str] = None
        cur_wh: Optional[Tuple[int, int]] = None
        for l in lines:
            if l.startswith("### IMAGE:"):
                cur_base = os.path.splitext(l.split("### IMAGE:", 1)[1].strip())[0]
                cur_path = getattr(state, "basename_to_path", {}).get(cur_base)
                cur_wh = _get_image_size_cached(state, cur_path) if cur_path else None
            elif not l.strip():
                continue
            else:
                if cur_base and cur_path and cur_wh:
                    b = _yolo_line_to_px(state, l, cur_path, wh=cur_wh)
                    if b:
                        mapping.setdefault(cur_base, []).append(b)
        return mapping

    # Fallback single-image: assume current file (if any)
    base = None
    cur_path = state.current_path() if hasattr(state, "current_path") else None
    if cur_path:
        base = os.path.splitext(os.path.basename(cur_path))[0]
    if base:
        boxes: List[Box] = []
        wh = _get_image_size_cached(state, cur_path)
        for l in lines:
            b = _yolo_line_to_px(state, l, cur_path, wh=wh)
            if b:
                boxes.append(b)
        if boxes:
            mapping[base] = boxes
    return mapping

def _quick_basename_index(state: AppState) -> Dict[str, Tuple[str, Optional[Tuple[int,int]]]]:
    """
    basename(lower) -> (full_path, (w,h) or None)
    - state.basename_to_path / _basename_to_path / image_files 로부터 구성
    - (w,h)는 필요 시 한번만 계산해서 캐시 사용
    """
    # 우선순위: basename_to_path -> _basename_to_path -> image_files
    b2p = getattr(state, "basename_to_path", None) or getattr(state, "_basename_to_path", None)
    if not b2p:
        files = getattr(state, "image_files", None) or []
        b2p = {os.path.splitext(os.path.basename(p))[0]: p for p in files}
        try:
            setattr(state, "basename_to_path", b2p)
        except Exception:
            pass

    idx: Dict[str, Tuple[str, Optional[Tuple[int,int]]]] = {}
    for base, p in b2p.items():
        idx[base.strip().lower()] = (p, None)

    # 가능하면 현재 보이는 이미지 사이즈는 바로 넣어줌(불필요 재계산 방지)
    if getattr(state, "img_w", 0) and getattr(state, "img_h", 0):
        cur = state.current_path()
        if cur:
            base = os.path.splitext(os.path.basename(cur))[0].strip().lower()
            idx[base] = (cur, (int(state.img_w), int(state.img_h)))

    return idx

def _load_csv_tsv_into_state(state: AppState, path: str) -> int:
    import csv
    delim = "\t" if path.lower().endswith(".tsv") else ","
    idx = _quick_basename_index(state)
    loaded = 0

    def _add(imgname: str, x1: str, y1: str, x2: str, y2: str, label: Optional[str]):
        nonlocal loaded
        base = os.path.splitext(os.path.basename(imgname))[0].strip().lower()
        if base not in idx:
            return
        full, wh = idx[base]
        try:
            bx = Box(float(x1), float(y1), float(x2), float(y2), (label or DEFAULT_CLASS_NAME), None)
        except Exception:
            return
        # 팔레트 보장
        if bx.label not in state.class_list:
            state.class_list.append(bx.label)
            state.class_colors[bx.label] = state.class_colors.get(bx.label, random_color())
        # 이미지별 클램프 (사이즈 없으면 캐시 조회)
        if wh is None:
            wh = _get_image_size_cached(state, full)
            # 한 번 채운 사이즈는 인덱스에 저장(다음 박스부터 재사용)
            b, _ = os.path.splitext(os.path.basename(full))
            idx[base] = (full, wh)
        if wh:
            iw, ih = wh
            bx = bx.clamp_to(iw, ih)
        state.annotations.setdefault(full, []).append(bx)
        loaded += 1

    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter=delim)
        rows_iter = (r for r in reader if any(c.strip() for c in r))

        # 헤더 판별 (첫 줄만 미리 뽑음)
        try:
            first = next(rows_iter)
        except StopIteration:
            return 0
        header = [c.strip().lower() for c in first]
        headered = any(not is_number(c) for c in header)

        if headered:
            h = {k: i for i, k in enumerate(header)}
            # dir,label,x1,y1,x2,y2
            if {"dir", "x1", "y1", "x2", "y2"} <= set(h.keys()):
                for r in rows_iter:
                    try:
                        imgname = r[h["dir"]]
                        x1, y1, x2, y2 = r[h["x1"]], r[h["y1"]], r[h["x2"]], r[h["y2"]]
                        label = r[h["label"]] if "label" in h and h["label"] < len(r) else DEFAULT_CLASS_NAME
                        _add(imgname, x1, y1, x2, y2, label)
                    except Exception:
                        continue
            # image_filename 계열
            elif "image_filename" in h:
                for r in rows_iter:
                    try:
                        imgname = r[h["image_filename"]]
                        def g(name, fallback_idx):
                            return r[h[name]] if name in h and h[name] < len(r) else r[fallback_idx]
                        x1 = g("x1", 1) if "x1" in h or "xmin" not in h else r[h["xmin"]]
                        y1 = g("y1", 2) if "y1" in h or "ymin" not in h else r[h["ymin"]]
                        x2 = g("x2", 3) if "x2" in h or "xmax" not in h else r[h["xmax"]]
                        y2 = g("y2", 4) if "y2" in h or "ymax" not in h else r[h["ymax"]]
                        label = r[h["label"]] if "label" in h and h["label"] < len(r) else DEFAULT_CLASS_NAME
                        _add(imgname, x1, y1, x2, y2, label)
                    except Exception:
                        continue
            else:
                # 추측 모드(헤더 있지만 케이스 불명)
                for r in rows_iter:
                    try:
                        # img, label, x1..y2
                        if len(r) >= 6 and (not is_number(r[0])) and all(is_number(c) for c in r[2:6]):
                            _add(r[0], r[2], r[3], r[4], r[5], r[1])
                        # x1..y2,label (파일명 없음 → 파일명은 CSV 파일명 기준)
                        elif len(r) >= 5 and all(is_number(c) for c in r[0:4]):
                            base = os.path.splitext(os.path.basename(path))[0]
                            _add(base, r[0], r[1], r[2], r[3], r[4])
                    except Exception:
                        continue
        else:
            # 헤더 없음: 숫자 모드
            base_file = os.path.splitext(os.path.basename(path))[0]
            for r in rows_iter:
                if len(r) >= 5 and all(is_number(c) for c in r[0:4]):
                    _add(base_file, r[0], r[1], r[2], r[3], r[4])

    return loaded

def _load_xml_into_state(state: AppState, path: str) -> int:
    """
    <annotations><image file="...">...</image>...</annotations>
    또는 Pascal VOC 단일 파일 케이스 모두 지원.
    메모리 전체 파싱 대신 iterparse 사용.
    """
    import xml.etree.ElementTree as ET
    idx = _quick_basename_index(state)
    loaded = 0

    def _flush_pascal(root):
        nonlocal loaded
        fname = (root.findtext("filename") or "").strip()
        if not fname:
            return
        base = os.path.splitext(os.path.basename(fname))[0].strip().lower()
        if base not in idx:
            return
        full, wh = idx[base]
        iw, ih = (None, None)
        if wh is None:
            wh = _get_image_size_cached(state, full)
            idx[base] = (full, wh)
        if wh:
            iw, ih = wh

        for e_obj in root.findall(".//object"):
            name = (e_obj.findtext("name") or DEFAULT_CLASS_NAME).strip()
            try:
                xmin = float(e_obj.findtext("bndbox/xmin"))
                ymin = float(e_obj.findtext("bndbox/ymin"))
                xmax = float(e_obj.findtext("bndbox/xmax"))
                ymax = float(e_obj.findtext("bndbox/ymax"))
            except Exception:
                continue
            bx = Box(xmin, ymin, xmax, ymax, name, None)
            if iw and ih:
                bx = bx.clamp_to(iw, ih)
            if bx.label not in state.class_list:
                state.class_list.append(bx.label)
                state.class_colors[bx.label] = state.class_colors.get(bx.label, random_color())
            state.annotations.setdefault(full, []).append(bx)
            loaded += 1

    # 먼저 루트 태그를 살짝 읽어 어떤 케이스인지 판단
    with open(path, "r", encoding="utf-8") as f:
        head = f.read(4096)
    agg_mode = ("<annotations" in head and "<image" in head)

    if agg_mode:
        # Aggregated: 이미지 단위로 이벤트 처리
        for event, elem in ET.iterparse(path, events=("end",)):
            if elem.tag == "image":
                file_attr = (elem.attrib.get("file", "") or "").strip()
                base = os.path.splitext(os.path.basename(file_attr))[0].strip().lower()
                if base in idx:
                    full, wh = idx[base]
                    if wh is None:
                        wh = _get_image_size_cached(state, full)
                        idx[base] = (full, wh)
                    iw, ih = (wh or (None, None))
                    for e_obj in elem.findall(".//object"):
                        name = (e_obj.findtext("name") or DEFAULT_CLASS_NAME).strip()
                        try:
                            xmin = float(e_obj.findtext("bndbox/xmin"))
                            ymin = float(e_obj.findtext("bndbox/ymin"))
                            xmax = float(e_obj.findtext("bndbox/xmax"))
                            ymax = float(e_obj.findtext("bndbox/ymax"))
                        except Exception:
                            continue
                        bx = Box(xmin, ymin, xmax, ymax, name, None)
                        if iw and ih:
                            bx = bx.clamp_to(iw, ih)
                        if bx.label not in state.class_list:
                            state.class_list.append(bx.label)
                            state.class_colors[bx.label] = state.class_colors.get(bx.label, random_color())
                        state.annotations.setdefault(full, []).append(bx)
                        loaded += 1
                elem.clear()  # 메모리 해제
        return loaded
    else:
        # Pascal VOC 단일 파일
        try:
            root = ET.parse(path).getroot()
        except Exception:
            return 0
        _flush_pascal(root)
        return loaded

def _load_yolo_txt_into_state(state: AppState, path: str) -> int:
    idx = _quick_basename_index(state)
    loaded = 0

    def add_for(base_key: str, line: str, full_path: str, wh: Optional[Tuple[int,int]]):
        nonlocal loaded
        b = _yolo_line_to_px(state, line, full_path, wh=wh)
        if not b:
            return
        if b.label not in state.class_list:
            state.class_list.append(b.label)
            state.class_colors[b.label] = state.class_colors.get(b.label, random_color())
        state.annotations.setdefault(full_path, []).append(b)
        loaded += 1

    with open(path, "r", encoding="utf-8") as f:
        lines = [l.rstrip("\n") for l in f]

    if any(l.startswith("### IMAGE:") for l in lines):
        cur_base = None
        cur_key = None
        cur_full = None
        cur_wh = None
        for l in lines:
            if l.startswith("### IMAGE:"):
                cur_base = os.path.splitext(l.split("### IMAGE:", 1)[1].strip())[0]
                cur_key = cur_base.strip().lower()
                if cur_key in idx:
                    cur_full, cur_wh = idx[cur_key]
                    if cur_wh is None:
                        cur_wh = _get_image_size_cached(state, cur_full)
                        idx[cur_key] = (cur_full, cur_wh)
                else:
                    cur_full, cur_wh = None, None
            elif not l.strip():
                continue
            else:
                if cur_full:
                    add_for(cur_key, l, cur_full, cur_wh)
        return loaded

    # 단일 이미지 가정: 현재 이미지 기준
    cur = state.current_path()
    if not cur:
        return 0
    base = os.path.splitext(os.path.basename(cur))[0].strip().lower()
    if base not in idx:
        return 0
    full, wh = idx[base]
    if wh is None:
        wh = _get_image_size_cached(state, full)
        idx[base] = (full, wh)

    for l in lines:
        add_for(base, l, full, wh)
    return loaded

def _load_json_into_state(state: AppState, path: str) -> int:
    # 기존 파서 논리 재사용(충분히 빠른 편). 필요시 NDJSON 별도 함수 추가 가능.
    mapping = _parse_json_file(state, path)
    return _merge_mapping_into_state(state, mapping)

def load_annotations(state: AppState, path: str) -> int:
    ext = os.path.splitext(path)[1].lower()
    if ext in (".csv", ".tsv"):
        return _load_csv_tsv_into_state(state, path)
    if ext == ".xml":
        return _load_xml_into_state(state, path)
    if ext == ".txt":
        return _load_yolo_txt_into_state(state, path)
    if ext == ".json":
        return _load_json_into_state(state, path)
    return 0


# --------------------------- Helpers ---------------------------

# omagari/io/annotation_io.py
import os  # <- 상단 import에 없으면 추가

def _merge_mapping_into_state(state: AppState, mapping: Dict[str, List[Box]]) -> int:
    """
    Merge parsed mapping {basename: [Box,...]} into state's annotations.
    Clamp per-target-image using its actual size (cache-backed).
    - basename_to_path / _basename_to_path 둘 다 지원
    - 없으면 image_files로 즉석 빌드
    """
    if not mapping:
        return 0

    # --- 안전한 basename 매핑 확보 ---
    b2p = getattr(state, "basename_to_path", None)
    if not b2p:
        b2p = getattr(state, "_basename_to_path", None)

    if not b2p or not isinstance(b2p, dict) or not b2p:
        # image_files에서 즉석 생성
        img_files = getattr(state, "image_files", None) or []
        b2p = {os.path.splitext(os.path.basename(p))[0]: p for p in img_files}
        # 가능하면 state에도 저장(다음에 재사용)
        try:
            setattr(state, "basename_to_path", b2p)
        except Exception:
            pass

    loaded = 0
    for base, boxes in mapping.items():
        img_path = b2p.get(base)
        if not img_path:
            # (확장자 대소문자/공백 문제 등) 느슨한 매칭 한 번 더 시도
            key = base.strip().lower()
            # 키 정규화 맵 만들기 (1회)
            if "__loose_b2p__" not in state.__dict__:
                try:
                    state.__dict__["__loose_b2p__"] = {
                        os.path.splitext(os.path.basename(p))[0].strip().lower(): p
                        for p in getattr(state, "image_files", []) or []
                    }
                except Exception:
                    state.__dict__["__loose_b2p__"] = {}
            img_path = state.__dict__["__loose_b2p__"].get(key)

        if not img_path:
            continue

        # ensure label palette
        for b in boxes:
            if b.label not in state.class_list:
                state.class_list.append(b.label)
                state.class_colors[b.label] = state.class_colors.get(b.label, random_color())

        # clamp per-image by its own dimensions
        w_h = _get_image_size_cached(state, img_path)
        if w_h is None:
            clamped = boxes
        else:
            iw, ih = w_h
            clamped = [b.clamp_to(iw, ih) for b in boxes]

        dst = state.annotations.setdefault(img_path, [])
        dst.extend(clamped)
        loaded += len(clamped)

    return loaded


def _get_image_size_cached(state: AppState, img_path: Optional[str]) -> Optional[Tuple[int, int]]:
    if not img_path:
        return None
    cache = getattr(state, "image_size_cache", None)
    if cache is not None and img_path in cache:
        return cache[img_path]
    try:
        from PIL import Image
        with Image.open(img_path) as im:
            w, h = im.size
    except Exception:
        try:
            import cv2
            im = cv2.imread(img_path)
            if im is None:
                return None
            h, w = im.shape[:2]
        except Exception:
            return None
    if cache is not None:
        cache[img_path] = (int(w), int(h))
    return (int(w), int(h))


def _yolo_line_to_px(state: AppState, line: str, img_path: str, wh: Optional[Tuple[int, int]] = None) -> Optional[Box]:
    parts = line.strip().split()
    if len(parts) < 5:
        return None
    try:
        cls_id = int(float(parts[0]))
        cx = float(parts[1]); cy = float(parts[2]); bw = float(parts[3]); bh = float(parts[4])
    except Exception:
        return None

    if wh is None:
        wh = _get_image_size_cached(state, img_path)
    if wh is None:
        return None
    w, h = wh

    x1 = (cx - bw / 2.0) * w
    y1 = (cy - bh / 2.0) * h
    x2 = (cx + bw / 2.0) * w
    y2 = (cy + bh / 2.0) * h

    # Resolve label name from class_list index when possible
    if 0 <= cls_id < len(state.class_list):
        label = state.class_list[cls_id]
    else:
        label = f"cls{cls_id}"
        if label not in state.class_list:
            state.class_list.append(label)
            state.class_colors[label] = state.class_colors.get(label, random_color())

    x1, y1, x2, y2 = normalize_box(x1, y1, x2, y2)
    return Box(x1, y1, x2, y2, label, None)
