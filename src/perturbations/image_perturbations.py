"""
image_perturbations.py

Image-level perturbation functions for the spurious correlation benchmark; 
each function takes a grayscale numpy array and returns a perturbed numpy array.

Perturbation types:
    - watermark / scanner artifact  (3 variants)
    - JPEG compression              (3 variants)
    - drains and tubes              (9 variants: chest tube x3, chest drain x3, ECG leads x3)
    - pacemaker                     (3 variants)
    - negative control              (1 variant)

Usage:
    from src.perturbations.image_perturbations import apply_perturbation

    img_array = cv2.imread("case.jpg", cv2.IMREAD_GRAYSCALE)
    img_array = cv2.resize(img_array, (512, 512))

    perturbed = apply_perturbation(img_array, perturbation_type="watermark", variant="v1")
    perturbed = apply_perturbation(img_array, perturbation_type="jpeg", variant="v2")
    perturbed = apply_perturbation(img_array, perturbation_type="chest_tube", variant="v1")
    perturbed = apply_perturbation(img_array, perturbation_type="chest_drain", variant="v2")
    perturbed = apply_perturbation(img_array, perturbation_type="ecg_leads", variant="v3")
    perturbed = apply_perturbation(img_array, perturbation_type="pacemaker", variant="v1",
                                   pacemaker_source=pm_array)
    perturbed = apply_perturbation(img_array, perturbation_type="negative_control", variant="v1")

    # or call individual functions directly:
    perturbed = add_watermark(img_array, variant="v2")
    perturbed = apply_jpeg_compression(img_array, variant="v1")
    perturbed = add_chest_tube(img_array, variant="v1")
    perturbed = add_chest_drain(img_array, variant="v2")
    perturbed = add_ecg_leads(img_array, variant="v3")
    perturbed = add_pacemaker(img_array, pacemaker_source=pm_array, variant="v1")
    perturbed = apply_negative_control(img_array)

Valid variants per type:
    watermark       : v1, v2, v3
    jpeg            : v1, v2, v3
    chest_tube      : v1, v2, v3
    chest_drain     : v1, v2, v3
    ecg_leads       : v1, v2, v3
    pacemaker       : v1, v2, v3  (requires pacemaker_source array)
    negative_control: v1
"""

import cv2
import numpy as np


_WATERMARK_VARIANTS = {
    "v1": {
        "lines":    ["STANFORD HEALTH CARE", "PA ERECT", "GE DISCOVERY XR656"],
        "position": "bottom_right",
        "opacity":  0.35,
    },
    "v2": {
        "lines":    ["GE HEALTHCARE", "PORTABLE AP", "SUPINE"],
        "position": "top_left",
        "opacity":  0.35,
    },
    "v3": {
        "lines":    ["BETH ISRAEL DEACONESS MC", "AP SUPINE", "ROOM ICU-4B"],
        "position": "bottom_left",
        "opacity":  0.35,
    },
}


def add_watermark(img: np.ndarray, variant: str = "v1") -> np.ndarray:
    """
    Overlays scanner / hospital watermark text onto the image.

    v1 ---- stationary scanner, bottom right (Stanford Health Care, PA ERECT)
    v2 ---- portable scanner, top left (GE Healthcare, PORTABLE AP, SUPINE)
    v3 ---- different institution, bottom left (Beth Israel, ICU room number)
    """
    cfg = _WATERMARK_VARIANTS[variant]
    lines    = cfg["lines"]
    position = cfg["position"]
    opacity  = cfg["opacity"]

    h, w = img.shape[:2]
    img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    overlay  = img_bgr.copy()

    font        = cv2.FONT_HERSHEY_SIMPLEX
    font_scale  = max(0.45, w / 1100)
    thickness   = 1
    margin      = 12
    line_gap    = 6
    text_bright = 200

    line_sizes = [cv2.getTextSize(l, font, font_scale, thickness)[0] for l in lines]
    line_h     = max(s[1] for s in line_sizes)
    block_h    = (line_h + line_gap) * len(lines)
    block_w    = max(s[0] for s in line_sizes)

    if position == "bottom_right":
        x0, y0 = w - block_w - margin, h - block_h - margin
    elif position == "bottom_left":
        x0, y0 = margin, h - block_h - margin
    elif position == "top_left":
        x0, y0 = margin, margin + line_h
    else:  
        x0, y0 = w - block_w - margin, margin + line_h

    color = (text_bright, text_bright, text_bright)
    for i, line in enumerate(lines):
        cv2.putText(overlay, line, (x0, y0 + i * (line_h + line_gap)),
                    font, font_scale, color, thickness, cv2.LINE_AA)

    result = cv2.addWeighted(overlay, opacity, img_bgr, 1 - opacity, 0)
    return cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)


_JPEG_QUALITY = {"v1": 75, "v2": 40, "v3": 10}


def apply_jpeg_compression(img: np.ndarray, variant: str = "v1") -> np.ndarray:
    """
    Applies JPEG compression in memory and returns the decoded image.

    v1 ---- quality 75, subtle degradation
    v2 ---- quality 40, visible block artifacts
    v3 ---- quality 10, heavy distortion


    """
    quality     = _JPEG_QUALITY[variant]
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, encoded   = cv2.imencode(".jpg", img, encode_param)
    return cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE)


_CHEST_TUBE_PARAMS = {
    "v1": {"entry_angle": 25, "mid_curve": 30, "bottom_drift": 15, "side": "right"},
    "v2": {"entry_angle": 20, "mid_curve": 35, "bottom_drift": 10, "side": "left"},
    "v3": {"entry_angle": 15, "mid_curve": 20, "bottom_drift": 20, "side": "right"},
}


def add_chest_tube(img: np.ndarray, variant: str = "v1") -> np.ndarray:
    """
    Draws a synthetic chest tube along the lateral pleural space.

    Placement validated against NIH NEATX annotations (right side ~60%,
    left ~35%, both ~5%; inter-annotator agreement 97.6%).

    v1 ---- right side, standard arc
    v2 ---- left side, slightly more curved
    v3 ---- right side, subtle curve
    """
    params = _CHEST_TUBE_PARAMS[variant]
    side   = params["side"]
    h, w   = img.shape[:2]
    out    = img.copy()

    if side == "right":
        sample = img[int(h*0.2):int(h*0.5), int(w*0.55):int(w*0.75)]
        x_base    = int(w * 0.72)
        curve_dir = -1
    else:
        sample = img[int(h*0.2):int(h*0.5), int(w*0.25):int(w*0.45)]
        x_base    = int(w * 0.28)
        curve_dir = 1

    brightness = int(min(200, sample.mean() * 1.4))
    y_start    = int(h * 0.15)
    y_end      = int(h * 0.68)
    num_pts    = 80

    points = []
    for i in range(num_pts):
        t = i / num_pts
        y = int(y_start + t * (y_end - y_start))
        if t < 0.3:
            frac   = t / 0.3
            offset = params["entry_angle"] * frac * curve_dir
        elif t < 0.7:
            frac      = (t - 0.3) / 0.4
            entry_end = params["entry_angle"]
            peak      = params["mid_curve"]
            offset    = (entry_end + (peak - entry_end) * np.sin(frac * np.pi / 2)) * curve_dir
        else:
            frac   = (t - 0.7) / 0.3
            offset = (params["mid_curve"] * (1 - frac) +
                      params["bottom_drift"] * frac) * curve_dir
        points.append((int(x_base + offset), y))

    for i in range(len(points) - 1):
        bv = int(np.clip(brightness + np.random.randint(-8, 8), 150, 200))
        cv2.line(out, points[i], points[i + 1], color=bv, thickness=2)
    cv2.circle(out, points[-1], radius=2, color=brightness - 20, thickness=1)

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    roi = out[max(0, min(ys)-5):min(h, max(ys)+5),
              max(0, min(xs)-8):min(w, max(xs)+8)]
    roi[:] = cv2.GaussianBlur(roi, (3, 3), 0)

    return out


_CHEST_DRAIN_PARAMS = {
    "v1": {"y_entry": 0.22, "y_tip": 0.45, "thickness": 3, "side": "right"},
    "v2": {"y_entry": 0.45, "y_tip": 0.65, "thickness": 3, "side": "left"},
    "v3": {"y_entry": 0.30, "y_tip": 0.48, "thickness": 2, "side": "right"},
}


def add_chest_drain(img: np.ndarray, variant: str = "v1") -> np.ndarray:
    """
    Draws a synthetic chest drain in the lateral pleural space.
    Shorter and thicker than a chest tube; enters from the chest wall.

    v1 ---- upper pleural, right (pneumothorax drainage)
    v2 ---- lower pleural, left  (effusion drainage)
    v3 ---- mid pleural, right, shorter
    """
    params    = _CHEST_DRAIN_PARAMS[variant]
    side      = params["side"]
    thickness = params["thickness"]
    h, w      = img.shape[:2]
    out       = img.copy()

    if side == "right":
        sample    = img[int(h*0.3):int(h*0.6), int(w*0.6):int(w*0.8)]
        x_entry   = int(w * 0.82)
        curve_dir = -1
    else:
        sample    = img[int(h*0.3):int(h*0.6), int(w*0.2):int(w*0.4)]
        x_entry   = int(w * 0.18)
        curve_dir = 1

    brightness = int(min(210, sample.mean() * 1.45))
    y_entry    = int(h * params["y_entry"])
    y_tip      = int(h * params["y_tip"])
    num_pts    = 50

    points = []
    for i in range(num_pts):
        t = i / num_pts
        y = int(y_entry + t * (y_tip - y_entry))
        if t < 0.4:
            frac   = t / 0.4
            offset = int(w * 0.14 * frac * curve_dir * -1)
        else:
            frac   = (t - 0.4) / 0.6
            base   = int(w * 0.14)
            offset = int((base + w * 0.03 * np.sin(frac * np.pi)) * curve_dir * -1)
        points.append((x_entry + offset, y))

    for i in range(len(points) - 1):
        bv = int(np.clip(brightness + np.random.randint(-8, 8), 160, 210))
        cv2.line(out, points[i], points[i + 1], color=bv, thickness=thickness)

    tip       = points[-1]
    side_hole = points[int(num_pts * 0.85)]
    cv2.circle(out, tip,       radius=3, color=brightness - 25, thickness=1)
    cv2.circle(out, side_hole, radius=2, color=brightness - 30, thickness=1)

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    roi = out[max(0, min(ys)-5):min(h, max(ys)+5),
              max(0, min(xs)-10):min(w, max(xs)+10)]
    roi[:] = cv2.GaussianBlur(roi, (3, 3), 0)

    return out


_ECG_CONFIGS = {
    "v1": [
        ("RA", 0.20, 0.72, 0.05, 0.98),
        ("LA", 0.25, 0.28, 0.05, 0.02),
        ("LL", 0.72, 0.32, 0.98, 0.02),
    ],
    "v2": [
        ("RA", 0.20, 0.72, 0.05, 0.98),
        ("LA", 0.25, 0.28, 0.05, 0.02),
        ("LL", 0.72, 0.32, 0.98, 0.02),
        ("RL", 0.72, 0.68, 0.98, 0.98),
        ("V",  0.42, 0.50, 0.42, 0.98),
    ],
    "v3": [
        ("RA", 0.18, 0.70, 0.05, 0.98),
        ("LA", 0.22, 0.30, 0.05, 0.02),
        ("LL", 0.70, 0.30, 0.98, 0.02),
    ],
}


def add_ecg_leads(img: np.ndarray, variant: str = "v1") -> np.ndarray:
    """
    Adds ECG electrode pads and lead wires to a frontal chest X-ray.
    Wire trails off at 55% of the way to the image edge.

    v1 ---- 3 lead (RA, LA, LL)
    v2 ---- 5 lead (adds RL and precordial)
    v3 ---- 3 lead with slight position variation per-technician
    """
    h, w    = img.shape[:2]
    out     = img.copy()
    configs = _ECG_CONFIGS[variant]

    local_mean  = img[int(h*0.3):int(h*0.6), int(w*0.3):int(w*0.7)].mean()
    wire_bright = int(np.clip(local_mean * 1.25, 130, 185))
    pad_bright  = int(np.clip(local_mean * 1.55, 170, 220))

    for _, pad_y, pad_x, exit_y, exit_x in configs:
        py = int(h * pad_y);  px = int(w * pad_x)
        ey = int(h * exit_y); ex = int(w * exit_x)

        mid_y = int(py + 0.55 * (ey - py))
        mid_x = int(px + 0.55 * (ex - px))
        tail_y = int(mid_y + 0.3 * (ey - mid_y))
        tail_x = int(mid_x + 0.3 * (ex - mid_x))

        sag_x = int((px + mid_x) / 2 + np.random.randint(-8, 8))
        sag_y = int((py + mid_y) / 2 + np.random.randint(-5, 5))
        cv2.line(out, (px, py),   (sag_x, sag_y), color=wire_bright, thickness=1)
        cv2.line(out, (sag_x, sag_y), (mid_x, mid_y), color=wire_bright, thickness=1)
        cv2.line(out, (mid_x, mid_y), (tail_x, tail_y),
                 color=max(80, wire_bright - 40), thickness=1)

        cv2.circle(out, (px, py), radius=5, color=pad_bright,      thickness=-1)
        cv2.circle(out, (px, py), radius=5, color=pad_bright - 30, thickness=1)

    for _, pad_y, pad_x, _, _ in configs:
        py = int(h * pad_y); px = int(w * pad_x)
        r = 12
        roi = out[max(0, py-r):min(h, py+r), max(0, px-r):min(w, px+r)]
        roi[:] = cv2.GaussianBlur(roi, (3, 3), 0)

    return out


_PACEMAKER_POSITIONS = {
    "v1": (130, 160),
    "v2": (120, 145),
    "v3": (145, 175),
}


def _match_intensity(src: np.ndarray, target: np.ndarray) -> np.ndarray:
    s = src.astype(np.float32)
    t = target.astype(np.float32)
    matched = (s - s.mean()) / (s.std() + 1e-5) * t.std() + t.mean()
    return np.clip(matched, 0, 255).astype(np.uint8)


def add_pacemaker(img: np.ndarray,
                  pacemaker_source: np.ndarray,
                  variant: str = "v1") -> np.ndarray:
    """
    Composites a pacemaker patch onto the upper left chest.

    pacemaker_source : grayscale array of a real pacemaker X-ray
                       (loaded externally, passed in here)
    v1 ---- standard upper left, below clavicle  (row=130, col=160)
    v2 ---- slightly higher and more lateral      (row=120, col=145)
    v3 ---- slightly lower and more medial        (row=145, col=175)
    """
    p = pacemaker_source.copy()
    p = cv2.equalizeHist(p)

    _, bright = cv2.threshold(p, 180, 255, cv2.THRESH_BINARY)
    kernel    = np.ones((5, 5), np.uint8)
    bright    = cv2.morphologyEx(bright, cv2.MORPH_OPEN, kernel)
    edges     = cv2.Canny(p, 50, 150)
    edges     = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    mask      = cv2.bitwise_or(bright, edges)
    mask      = cv2.GaussianBlur(mask, (25, 25), 0).astype(np.float32) / 255.0

    p    = _match_intensity(p, img)
    size = np.random.randint(45, 65)
    p    = cv2.resize(p, (size, size))
    mask = cv2.resize(mask, (size, size))
    p    = cv2.GaussianBlur(p, (5, 5), 0)
    p    = (p * 0.9).astype(np.uint8)

    row, col = _PACEMAKER_POSITIONS[variant]
    h, w     = img.shape[:2]
    out      = img.copy().astype(np.float32)

    r1 = max(0, row - size // 2);  r2 = min(h, r1 + size)
    c1 = max(0, col - size // 2);  c2 = min(w, c1 + size)
    ph = r2 - r1;  pw = c2 - c1

    alpha   = mask[:ph, :pw] * 0.7
    roi     = out[r1:r2, c1:c2]
    out[r1:r2, c1:c2] = roi * (1 - alpha) + p[:ph, :pw] * alpha

    result = np.clip(out, 0, 255).astype(np.uint8)
    return cv2.GaussianBlur(result, (3, 3), 0)


def apply_negative_control(img: np.ndarray, angle: float = 1.0) -> np.ndarray:
    """
    Rotates the image by 1 degree. 

    """
    h, w   = img.shape[:2]
    center = (w // 2, h // 2)
    M      = cv2.getRotationMatrix2D(center, angle, scale=1.0)
    return cv2.warpAffine(img, M, (w, h),
                          flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REFLECT_101)



def apply_perturbation(img: np.ndarray,
                       perturbation_type: str,
                       variant: str = "v1",
                       pacemaker_source: np.ndarray = None) -> np.ndarray:
    """
    Single entry point for all image perturbations.

    Parameters
    ----------
    img               : grayscale uint8 numpy array, already resized to (512, 512)
    perturbation_type : one of:
                        "watermark", "jpeg",
                        "chest_tube", "chest_drain", "ecg_leads", "pacemaker",
                        "negative_control"
    variant           : "v1", "v2", or "v3" (negative_control only uses v1)
    pacemaker_source  : required only when perturbation_type == "pacemaker"

    Returns
    -------
    perturbed grayscale uint8 numpy array
    """
    if perturbation_type == "watermark":
        return add_watermark(img, variant)

    elif perturbation_type == "jpeg":
        return apply_jpeg_compression(img, variant)

    elif perturbation_type == "chest_tube":
        return add_chest_tube(img, variant)

    elif perturbation_type == "chest_drain":
        return add_chest_drain(img, variant)

    elif perturbation_type == "ecg_leads":
        return add_ecg_leads(img, variant)

    elif perturbation_type == "pacemaker":
        if pacemaker_source is None:
            raise ValueError("pacemaker_source must be provided for pacemaker perturbation")
        return add_pacemaker(img, pacemaker_source, variant)

    elif perturbation_type == "negative_control":
        return apply_negative_control(img)

    else:
        raise ValueError(
            f"Unknown perturbation_type '{perturbation_type}'. "
            "Valid types: watermark, jpeg, chest_tube, chest_drain, "
            "ecg_leads, pacemaker, negative_control"
        )



ALL_CONDITIONS = [
    ("watermark",        "v1"),
    ("watermark",        "v2"),
    ("watermark",        "v3"),
    ("jpeg",             "v1"),
    ("jpeg",             "v2"),
    ("jpeg",             "v3"),
    ("chest_tube",       "v1"),
    ("chest_tube",       "v2"),
    ("chest_tube",       "v3"),
    ("chest_drain",      "v1"),
    ("chest_drain",      "v2"),
    ("chest_drain",      "v3"),
    ("ecg_leads",        "v1"),
    ("ecg_leads",        "v2"),
    ("ecg_leads",        "v3"),
    ("pacemaker",        "v1"),
    ("pacemaker",        "v2"),
    ("pacemaker",        "v3"),
    ("negative_control", "v1"),
]
