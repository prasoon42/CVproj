# src/detect_and_ocr.py
import os
import time
import re
import cv2
import numpy as np
import easyocr
from src.utils import group_boxes_by_row

# ----------------- Helpers: IoU / NMS / merging -----------------
def iou(a,b):
    xA = max(a[0], b[0])
    yA = max(a[1], b[1])
    xB = min(a[2], b[2])
    yB = min(a[3], b[3])
    interW = max(0, xB-xA)
    interH = max(0, yB-yA)
    inter = interW * interH
    areaA = max(0,(a[2]-a[0]))*max(0,(a[3]-a[1]))
    areaB = max(0,(b[2]-b[0]))*max(0,(b[3]-b[1]))
    union = areaA + areaB - inter
    if union<=0:
        return 0.0
    return inter/union

def nms_boxes(boxes, iou_thresh=0.35):
    if not boxes:
        return []
    boxes = sorted(boxes, key=lambda x: x[4], reverse=True)
    keep = []
    used = [False]*len(boxes)
    for i,b in enumerate(boxes):
        if used[i]:
            continue
        keep.append(b)
        for j in range(i+1,len(boxes)):
            if used[j]:
                continue
            if iou(b, boxes[j]) > iou_thresh:
                used[j] = True
    return keep

def merge_close_boxes_row(boxes_row, gap_threshold_frac=0.16):
    """
    Merge boxes that are close horizontally. less aggressive to avoid collapsing digits.
    """
    if not boxes_row:
        return []
    merged = []
    widths = [max(1,b[2]-b[0]) for b in boxes_row]
    mean_w = max(1.0, float(sum(widths))/len(widths))
    gap_thresh = mean_w * (1.0 + gap_threshold_frac)
    cur = boxes_row[0].copy()
    for b in boxes_row[1:]:
        gap = b[0] - cur[2]
        if gap <= gap_thresh:
            cur[2] = max(cur[2], b[2])
            cur[3] = max(cur[3], b[3])
            cur[0] = min(cur[0], b[0])
            cur[1] = min(cur[1], b[1])
            cur[4] = max(cur[4], b[4])
        else:
            merged.append(cur)
            cur = b.copy()
    merged.append(cur)
    return merged

# ----------------- Roboflow / OCR SETUP -----------------
api_key = os.environ.get("ROBOFLOW_API_KEY")
CLIENT = None
if api_key:
    try:
        from inference_sdk import InferenceHTTPClient
        CLIENT = InferenceHTTPClient(api_url="https://serverless.roboflow.com", api_key=api_key)
        print("[init] Roboflow client initialized.", flush=True)
    except Exception as e:
        print("[WARN] Could not init Roboflow client:", e, flush=True)
        CLIENT = None
else:
    print("[WARN] ROBOFLOW_API_KEY not set. Roboflow inference will be skipped (local fallback will be used).", flush=True)

MODEL_ID = "7-segment-display-gxhnj/2"

# EasyOCR reader (CPU on Mac)
try:
    reader = easyocr.Reader(['en'], gpu=False)
    print("[init] EasyOCR reader ready.", flush=True)
except Exception as e:
    raise RuntimeError("Install easyocr and torch properly. Error: "+str(e))

# ----------------- PREPROCESS -----------------
def preprocess_image(img, max_side=1000):
    if img is None:
        return None
    h, w = img.shape[:2]
    scale = max_side / max(h, w)
    if scale < 1:
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
    return img

# ----------------- OCR helpers: vertical split -----------------
def vertical_split_segments(bin_img, min_width=6, valley_frac=0.10):
    """
    Column projection segmentation; returns list of (x1,x2)
    """
    h,w = bin_img.shape[:2]
    col_sum = np.sum(bin_img==255, axis=0)
    max_sum = col_sum.max() if col_sum.size>0 else 0
    if max_sum == 0:
        return [(0,w)]
    valley_thresh = max(1, int(max_sum * valley_frac))
    segments = []
    in_seg = False
    seg_start = 0
    for x in range(w):
        if col_sum[x] > valley_thresh:
            if not in_seg:
                in_seg = True
                seg_start = x
        else:
            if in_seg:
                seg_end = x
                if seg_end - seg_start >= min_width:
                    segments.append((seg_start, seg_end))
                in_seg = False
    if in_seg:
        seg_end = w
        if seg_end - seg_start >= min_width:
            segments.append((seg_start, seg_end))
    if not segments:
        return [(0,w)]
    # merge tiny neighbors
    merged = []
    prev = segments[0]
    for s in segments[1:]:
        if s[0] - prev[1] <= 2:
            prev = (prev[0], s[1])
        else:
            merged.append(prev)
            prev = s
    merged.append(prev)
    return merged

# ----------------- OCR core (improved) -----------------
def ocr_crop(img, box):
    """
    Improved OCR for a crop. Accepts either (full-image, box) or (crop, [0,0,Wc,Hc]).
    This function:
      - builds multiple preprocessed candidate images (adaptive, otsu, inverted, etc.)
      - tries OCR on those candidates
      - if crop is wide and segmentation fails to split it, forcibly split into N equal vertical slices
      - if normal candidates fail to detect thin strokes, runs Sobel (edge) image OCR as fallback
    """
    import math
    def pad_and_resize_crop_local(img_local, box_local, pad_frac=0.35, min_side=72):
        h, w = img_local.shape[:2]
        x1, y1, x2, y2 = box_local[:4]
        bw = max(1, x2 - x1)
        bh = max(1, y2 - y1)
        padx = int(bw * pad_frac)
        pady = int(bh * pad_frac)
        x1s = max(0, x1 - padx)
        y1s = max(0, y1 - pady)
        x2s = min(w, x2 + padx)
        y2s = min(h, y2 + pady)
        crop = img_local[y1s:y2s, x1s:x2s]
        if crop.size == 0:
            return None
        short_side = min(crop.shape[0], crop.shape[1])
        if short_side < min_side:
            scale = float(min_side) / float(short_side)
            crop = cv2.resize(crop, (int(crop.shape[1]*scale), int(crop.shape[0]*scale)), interpolation=cv2.INTER_CUBIC)
        return crop

    # If user passed a crop already (0..Wc), handle that:
    try:
        # if box corresponds to full-crop coords (0..Wc), treat img as crop
        if box[0] == 0 and box[1] == 0:
            crop = img
        else:
            crop = pad_and_resize_crop_local(img, box, pad_frac=0.35, min_side=72)
    except Exception:
        # fallback simple crop attempt
        crop = pad_and_resize_crop_local(img, box, pad_frac=0.35, min_side=72)

    if crop is None or crop.size == 0:
        return ''

    Hc, Wc = crop.shape[:2]

    # Build candidate images
    try:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    except Exception:
        gray = crop if len(crop.shape)==2 else cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    candidates = []
    candidates.append(('color', crop.copy()))
    try:
        th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 4)
        candidates.append(('adap', th))
    except Exception:
        pass
    try:
        _, ots = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        candidates.append(('otsu', ots))
    except Exception:
        pass
    try:
        inv = cv2.bitwise_not(gray)
        _, inv_th = cv2.threshold(inv, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        candidates.append(('inv_otsu', inv_th))
    except Exception:
        pass
    try:
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        closed = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, k)
        _, closed_th = cv2.threshold(closed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        candidates.append(('closed', closed_th))
    except Exception:
        pass

    # OCR all candidates and collect digit-only results with confidences
    ocr_results = []
    for name, cand in candidates:
        try:
            res = reader.readtext(cand, detail=1)
            for bbox, text, conf in res:
                if not text:
                    continue
                digits = ''.join(ch for ch in (text or '') if ch.isdigit())
                if digits == '':
                    continue
                ocr_results.append({'source': name, 'text_raw': text, 'digits': digits, 'conf': float(conf)})
        except Exception:
            continue

    print("[ocr_debug] crop size:", (Wc, Hc), "candidates:", [(r['digits'], round(r['conf'],2), r['source']) for r in ocr_results], flush=True)

    # pick best direct candidate if it looks plausible
    best = None
    if ocr_results:
        # prefer 2-3 digit results with high conf
        sorted_results = sorted(ocr_results, key=lambda x: x['conf'], reverse=True)
        for r in sorted_results:
            if 1 <= len(r['digits']) <= 3:
                best = r
                break
        if best is None:
            best = sorted_results[0]

    # segmentation attempt using vertical projection
    def vertical_split_segments_local(bin_img, min_width=6, valley_frac=0.10):
        col_sum = np.sum(bin_img==255, axis=0)
        if col_sum.size==0:
            return [(0, bin_img.shape[1])]
        max_sum = col_sum.max()
        valley_thresh = max(1, int(max_sum * valley_frac))
        segments = []
        in_seg = False
        seg_start = 0
        for x in range(col_sum.shape[0]):
            if col_sum[x] > valley_thresh:
                if not in_seg:
                    in_seg = True
                    seg_start = x
            else:
                if in_seg:
                    seg_end = x
                    if seg_end - seg_start >= min_width:
                        segments.append((seg_start, seg_end))
                    in_seg = False
        if in_seg:
            seg_end = col_sum.shape[0]
            if seg_end - seg_start >= min_width:
                segments.append((seg_start, seg_end))
        if not segments:
            return [(0, bin_img.shape[1])]
        # merge small neighbors
        merged = []
        prev = segments[0]
        for s in segments[1:]:
            if s[0] - prev[1] <= 2:
                prev = (prev[0], s[1])
            else:
                merged.append(prev)
                prev = s
        merged.append(prev)
        return merged

    # compute a binary for segmentation (prefer Otsu or adaptive)
    try:
        _, bin_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    except Exception:
        bin_img = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 4)

    # invert if needed so digits are white on black projection
    white_frac = np.sum(bin_img==255) / float(bin_img.size)
    if white_frac < 0.5:
        bin_img = cv2.bitwise_not(bin_img)

    segs = vertical_split_segments_local(bin_img, min_width=6, valley_frac=0.10)

    # If segmentation produced multiple segments, OCR each and join
    if len(segs) > 1:
        seg_texts = []
        for (sx, ex) in segs:
            sub = crop[:, sx:ex]
            if sub.size == 0:
                continue
            try:
                subres = reader.readtext(sub, detail=1)
                best_sub = None
                for _, t, c_ in subres:
                    digits_sub = ''.join(ch for ch in (t or '') if ch.isdigit())
                    if not digits_sub:
                        continue
                    if best_sub is None or float(c_) > best_sub[1]:
                        best_sub = (digits_sub, float(c_))
                if best_sub:
                    seg_texts.append(best_sub[0])
            except Exception:
                continue
        if seg_texts:
            final = ''.join(seg_texts)
            final = re.sub(r'[^0-9]', '', final)
            print("[ocr_debug] segmented final:", final, "segs:", segs, flush=True)
            return final

    # If segmentation returned a single segment but crop is wide, force equal splits
    width_height_ratio = float(Wc) / max(1.0, Hc)
    if len(segs) == 1 and width_height_ratio > 1.6:
        # choose number of splits = round(ratio) but clamp 2..4
        n_splits = int(min(4, max(2, round(width_height_ratio))))
        split_texts = []
        for i in range(n_splits):
            sx = int(i * (Wc / n_splits))
            ex = int((i+1) * (Wc / n_splits))
            sub = crop[:, sx:ex]
            if sub.size == 0:
                continue
            try:
                subres = reader.readtext(sub, detail=1)
                best_sub = None
                for _, t, c_ in subres:
                    digits_sub = ''.join(ch for ch in (t or '') if ch.isdigit())
                    if not digits_sub:
                        continue
                    if best_sub is None or float(c_) > best_sub[1]:
                        best_sub = (digits_sub, float(c_))
                if best_sub:
                    split_texts.append(best_sub[0])
            except Exception:
                continue
        if split_texts:
            final = ''.join(split_texts)
            final = re.sub(r'[^0-9]', '', final)
            print("[ocr_debug] forced-split final:", final, "n_splits:", n_splits, flush=True)
            return final

    # If we have a best candidate from earlier, return it
    if best:
        return re.sub(r'[^0-9]', '', best['digits'])

    # Edge-preserving fallback: Sobel (preserves thin vertical strokes)
    try:
        gx = cv2.Sobel(gray, cv2.CV_16S, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_16S, 0, 1, ksize=3)
        g = cv2.magnitude(gx.astype(np.float32), gy.astype(np.float32))
        g = np.uint8(np.clip((g / g.max())*255, 0, 255))
        # threshold gradient image
        _, gth = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # try OCR on gradient
        try:
            gres = reader.readtext(gth, detail=1)
            best_g = None
            for _, t, c_ in gres:
                digits_g = ''.join(ch for ch in (t or '') if ch.isdigit())
                if digits_g:
                    if best_g is None or float(c_) > best_g[1]:
                        best_g = (digits_g, float(c_))
            if best_g:
                final = re.sub(r'[^0-9]', '', best_g[0])
                print("[ocr_debug] sobel fallback ->", final, flush=True)
                return final
        except Exception:
            pass
    except Exception:
        pass

    # final fallback: try reader.readtext simple (detail=0)
    try:
        fb = reader.readtext(crop, detail=0)
        if fb:
            txt = ''.join(ch for ch in fb[0] if ch.isdigit())
            print("[ocr_debug] fallback simple ->", txt, flush=True)
            return txt
    except Exception:
        pass

    return ''

# ----------------- Roboflow inference wrapper -----------------
def _roboflow_infer(img_rgb):
    if CLIENT is None:
        raise RuntimeError("Roboflow CLIENT not initialized")
    return CLIENT.infer(img_rgb, model_id=MODEL_ID)

def detect_digits_roboflow(img, max_retries=2, retry_delay=1.0):
    if CLIENT is None:
        return []
    if img is None:
        return []
    H, W = img.shape[:2]
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    attempt = 0
    last_exc = None
    while attempt <= max_retries:
        try:
            print(f"[Roboflow] sending inference request (attempt {attempt+1})...", flush=True)
            results = _roboflow_infer(img_rgb)
            preds = results.get("predictions", []) or []
            boxes = []
            for p in preds:
                if all(k in p for k in ("x","y","width","height")):
                    cx, cy, bw, bh = p["x"], p["y"], p["width"], p["height"]
                    x1 = int(cx - bw/2); y1 = int(cy - bh/2)
                    x2 = int(cx + bw/2); y2 = int(cy + bh/2)
                elif "bbox" in p:
                    x1,y1,x2,y2 = map(int, p["bbox"])
                else:
                    continue
                x1 = max(0, min(x1, W-1)); x2 = max(0, min(x2, W-1))
                y1 = max(0, min(y1, H-1)); y2 = max(0, min(y2, H-1))
                conf = float(p.get("confidence", p.get("score", 0)))
                cls = p.get("class", 0)
                try:
                    cls = int(cls)
                except Exception:
                    pass
                boxes.append([x1,y1,x2,y2,conf,cls])
            print(f"[Roboflow] returned {len(boxes)} boxes", flush=True)
            return boxes
        except Exception as e:
            last_exc = e
            msg = str(e)
            print(f"[Roboflow] inference exception: {msg}", flush=True)
            if "401" in msg or "Unauthorized" in msg or "Unauthorized api_key" in msg:
                print("[Roboflow] unauthorized API key or access denied - aborting Roboflow attempts.", flush=True)
                break
            attempt += 1
            if attempt <= max_retries:
                print(f"[Roboflow] retrying in {retry_delay}s...", flush=True)
                time.sleep(retry_delay)
    print("[Roboflow] failed after retries; error:", last_exc, flush=True)
    return []

# ----------------- Local YOLO fallback -----------------
def local_yolo_detect(img, conf_thres=0.35, min_area=200):
    try:
        from ultralytics import YOLO
    except Exception:
        print("[YOLO fallback] ultralytics not installed; skipping local YOLO fallback.", flush=True)
        return []
    weights_path = os.path.join(os.path.dirname(__file__), "..", "weights", "yolov8n.pt")
    if not os.path.exists(weights_path):
        weights_path = "yolov8n.pt"
    try:
        print(f"[YOLO fallback] loading weights {weights_path} ...", flush=True)
        model = YOLO(weights_path)
        res = model.predict(img, imgsz=640, conf=conf_thres, verbose=False)[0]
        boxes = []
        H, W = img.shape[:2]
        for b in res.boxes:
            try:
                xy = b.xyxy[0].tolist()
            except Exception:
                continue
            x1,y1,x2,y2 = map(int, xy)
            conf = float(getattr(b, "conf", [0.0])[0]) if hasattr(b, "conf") else 0.0
            cls = int(getattr(b, "cls", [0])[0]) if hasattr(b, "cls") else 0
            area = (x2-x1)*(y2-y1)
            if area < min_area:
                continue
            boxes.append([x1,y1,x2,y2,conf,cls])
        print(f"[YOLO fallback] returned {len(boxes)} boxes", flush=True)
        return boxes
    except Exception as e:
        print("[YOLO fallback] error:", e, flush=True)
        return []

# ----------------- High-level detect and postprocess -----------------
def detect_digits_raw(img):
    boxes = detect_digits_roboflow(img)
    if boxes:
        return boxes
    print("[detect_digits_raw] Roboflow empty or failed; trying local YOLO...", flush=True)
    boxes = local_yolo_detect(img)
    return boxes or []

def postprocess_boxes(raw_boxes, min_conf=0.45, min_area=600, iou_thresh=0.35):
    if not raw_boxes:
        return []
    filtered = []
    for b in raw_boxes:
        x1,y1,x2,y2,conf,cls = b
        w = max(0, x2-x1); h = max(0, y2-y1)
        area = w*h
        if conf < min_conf:
            continue
        if area < min_area:
            continue
        filtered.append(b)
    filtered_nms = nms_boxes(filtered, iou_thresh=iou_thresh)
    return filtered_nms

def detect_digits(img):
    raw = detect_digits_raw(img)
    boxes = postprocess_boxes(raw, min_conf=0.45, min_area=600, iou_thresh=0.35)
    print(f"[detect_digits] postprocessed -> {len(boxes)} boxes", flush=True)
    return boxes

# ----------------- Reconstruction & heuristics -----------------
def reconstruct_reading(img, boxes):
    if not boxes:
        return ''
    rows = group_boxes_by_row(boxes, threshold=30)
    rows = sorted(rows, key=lambda r: r['cy'])
    row_texts = []
    for r in rows:
        row_boxes = sorted(r['boxes'], key=lambda b: b[0])
        merged = merge_close_boxes_row(row_boxes, gap_threshold_frac=0.16)
        texts = []
        H,W = img.shape[:2]
        for m in merged:
            x1,y1,x2,y2 = m[:4]
            pad_x = int((x2-x1)*0.18)
            pad_y = int((y2-y1)*0.36)
            sx = max(0, x1-pad_x); sy = max(0, y1-pad_y)
            ex = min(W, x2+pad_x); ey = min(H, y2+pad_y)
            crop = img[sy:ey, sx:ex]
            if crop.size == 0:
                texts.append('')
                continue
            # call ocr_crop but pass crop coordinates (0..Wc)
            t = ocr_crop(crop, [0,0,crop.shape[1], crop.shape[0]])
            t = re.sub(r'[^0-9]', '', t)
            # if t is long, try splitting every 2-3 chars
            if len(t) > 3:
                # prefer splitting into plausible groups (e.g., 3 -> 1+2 or 2+1)
                # but for now keep full and let later logic split
                pass
            texts.append(t)
        texts = [x for x in texts if x.strip()!='']
        combined = ''.join(texts)
        row_texts.append(combined)
    # Heuristic flattening to plausible metrics
    plausible = []
    for t in row_texts:
        if not t:
            continue
        if 2 <= len(t) <= 3:
            plausible.append(t)
        elif len(t) == 4:
            plausible.append(t[:2]); plausible.append(t[2:])
        elif len(t) > 4:
            # split in chunks of 2 or 3 to try to recover (prefer 2 then 2 then ...)
            i=0
            while i < len(t):
                remain = len(t)-i
                take = 2 if remain%2==0 or remain==2 else 3
                plausible.append(t[i:i+take])
                i += take
    final = ' | '.join(plausible) if plausible else ' | '.join([r for r in row_texts if r])
    print("[reconstruct] row_texts:", row_texts, flush=True)
    print("[reconstruct] plausible:", plausible, flush=True)
    return final

def map_rows_to_metrics(img, boxes):
    rows = group_boxes_by_row(boxes, threshold=30)
    rows = sorted(rows, key=lambda r: r['cy'])
    values = []
    for r in rows:
        row_boxes = sorted(r['boxes'], key=lambda b: b[0])
        digits = [ocr_crop(img, b) for b in row_boxes]
        digits = [d for d in digits if d.strip()]
        values.append(''.join(digits))
    result = {}
    if len(values) > 0:
        result['SYS'] = values[0]
    if len(values) > 1:
        result['DIA'] = values[1]
    if len(values) > 2:
        result['PULSE'] = values[-1]
    return result

# ----------------- Visualization -----------------
def visualize(img, boxes, reading_str=None, show_conf=True):
    vis = img.copy()
    for b in boxes:
        x1,y1,x2,y2,conf,_ = b
        cv2.rectangle(vis, (x1,y1), (x2,y2), (0,255,0), 2)
        if show_conf:
            cv2.putText(vis, f"{conf:.2f}", (x1, max(0,y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1)
    if reading_str:
        cv2.putText(vis, f"Reading: {reading_str}", (10,35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2)
    return vis

# ----------------- CLI ENTRYPOINT -----------------
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python detect_and_ocr.py <image_path>")
        sys.exit(1)
    img_path = sys.argv[1]
    img = cv2.imread(img_path)
    if img is None:
        print("Could not read image:", img_path)
        sys.exit(1)
    img = preprocess_image(img)
    raw = detect_digits_raw(img)
    boxes = detect_digits(img)
    reading = reconstruct_reading(img, boxes)
    metrics = map_rows_to_metrics(img, boxes)
    vis = visualize(img, boxes, reading)
    out_path = "output.jpg"
    cv2.imwrite(out_path, vis)
    print("Detected Reading:", reading)
    print("Mapped metrics:", metrics)
    print("Saved visualization to", out_path)
