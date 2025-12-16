import cv2
import math
import numpy as np
from dataclasses import dataclass

from collections import Counter

def classify_chip(w):

    if w <= 45:
        return 1

    elif w > 45 and w < 100:
        return 2
    elif w >= 100:
        return 3


def figure_from_array(arr):
    counts = Counter(arr)

    c1 = counts.get(1, 0)
    c2 = counts.get(2, 0)
    c3 = counts.get(3, 0)

    # Only 1s
    if c2 == 0 and c3 == 0:
        return c1 - 1

    # One 2 and some 1s
    if c2 == 1 and c3 == 0:
        return 3 + c1

    # One 3 and some 1s
    if c3 == 1 and c2 == 0:
        return 6 + c1

    raise ValueError("Invalid combination")


def compute_output_board(dominant_color , figures):
    if dominant_color == "yellow":
        return int("".join(map(str, figures))) * 10
    elif dominant_color == "blue":
        return int("".join(map(str, figures)))
    elif dominant_color == "red":

        return math.prod(figures)
    

@dataclass
class CardDetectionResult:
    box: np.ndarray
    size: tuple
    warped: np.ndarray
    mask: np.ndarray

@dataclass
class DetectParams:
    blur_ksize: int = 5
    canny_lower: int = 50
    canny_upper: int = 150
    dark_s_max: int = 80
    dark_v_max: int = 90
    morph_kernel: int = 7
    morph_iter: int = 2
    area_min_percent: float = 1.0
    aspect_min: float = 2.0

def _order_points(pts: np.ndarray) -> np.ndarray:
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def _four_point_transform(image: np.ndarray, pts: np.ndarray) -> np.ndarray:
    rect = _order_points(pts)
    (tl, tr, br, bl) = rect
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = int(max(widthA, widthB))
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = int(max(heightA, heightB))
    dst = np.array([[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped

def _auto_canny(gray: np.ndarray) -> np.ndarray:
    med = np.median(gray)
    lower = int(max(0, 0.66 * med))
    upper = int(min(255, 1.33 * med))
    edges = cv2.Canny(gray, lower, upper)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges = cv2.dilate(edges, k, iterations=1)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, k, iterations=1)
    return edges

def detect_cards(image: np.ndarray) -> list[CardDetectionResult]:
    h, w = image.shape[:2]
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask_dark = cv2.inRange(hsv, (0, 0, 0), (180, 100, 120))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    closed = cv2.morphologyEx(mask_dark, cv2.MORPH_CLOSE, kernel, iterations=5)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = _auto_canny(blur)
    masked_edges = cv2.bitwise_and(edges, edges, mask=closed)
    cnts_edges, _ = cv2.findContours(masked_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    results = []
    for c in cnts_edges:
        area = cv2.contourArea(c)
        if area < (h * w) * 0.01:
            continue
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) != 4:
            continue
        box = approx.reshape(4, 2).astype(np.float32)
        d01 = np.linalg.norm(box[0] - box[1])
        d12 = np.linalg.norm(box[1] - box[2])
        aspect = max(d01, d12) / (min(d01, d12) + 1e-5)
        if aspect < 2.0:
            continue
        warped = _four_point_transform(image, box)
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [box.astype(np.int32)], 255)
        size = (warped.shape[1], warped.shape[0])
        results.append(CardDetectionResult(box=box, size=size, warped=warped, mask=mask))
    if not results:
        cnts_mask, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts_mask:
            area = cv2.contourArea(c)
            if area < (h * w) * 0.01:
                continue
            rect = cv2.minAreaRect(c)
            box = cv2.boxPoints(rect).astype(np.float32)
            warped = _four_point_transform(image, box)
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(mask, [box.astype(np.int32)], 255)
            size = (warped.shape[1], warped.shape[0])
            results.append(CardDetectionResult(box=box, size=size, warped=warped, mask=mask))
    results.sort(key=lambda r: r.size[0] * r.size[1], reverse=True)
    return results

def detect_cards_with_params(image: np.ndarray, params: DetectParams) -> tuple[list[CardDetectionResult], dict]:
    h, w = image.shape[:2]
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask_dark = cv2.inRange(hsv, (0, 0, 0), (180, int(params.dark_s_max), int(params.dark_v_max)))
    k = max(1, int(params.morph_kernel))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    closed = cv2.morphologyEx(mask_dark, cv2.MORPH_CLOSE, kernel, iterations=int(params.morph_iter))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    bk = max(1, int(params.blur_ksize))
    if bk % 2 == 0:
        bk += 1
    blur = cv2.GaussianBlur(gray, (bk, bk), 0)
    edges = cv2.Canny(blur, int(params.canny_lower), int(params.canny_upper))
    k3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges = cv2.dilate(edges, k3, iterations=1)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, k3, iterations=1)
    masked_edges = cv2.bitwise_and(edges, edges, mask=closed)
    cnts_edges, _ = cv2.findContours(masked_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    results = []
    min_area = (h * w) * (params.area_min_percent / 100.0)
    for c in cnts_edges:
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) != 4:
            continue
        box = approx.reshape(4, 2).astype(np.float32)
        d01 = np.linalg.norm(box[0] - box[1])
        d12 = np.linalg.norm(box[1] - box[2])
        aspect = max(d01, d12) / (min(d01, d12) + 1e-5)
        if aspect < params.aspect_min:
            continue
        warped = _four_point_transform(image, box)
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [box.astype(np.int32)], 255)
        size = (warped.shape[1], warped.shape[0])
        results.append(CardDetectionResult(box=box, size=size, warped=warped, mask=mask))
    if not results:
        cnts_mask, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts_mask:
            area = cv2.contourArea(c)
            if area < min_area:
                continue
            rect = cv2.minAreaRect(c)
            box = cv2.boxPoints(rect).astype(np.float32)
            warped = _four_point_transform(image, box)
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(mask, [box.astype(np.int32)], 255)
            size = (warped.shape[1], warped.shape[0])
            results.append(CardDetectionResult(box=box, size=size, warped=warped, mask=mask))
    results.sort(key=lambda r: r.size[0] * r.size[1], reverse=True)
    debug = {"mask_dark": mask_dark, "closed": closed, "blur": blur, "edges": edges, "masked_edges": masked_edges}
    return results, debug

def detect_cards_binary(image: np.ndarray) -> list[CardDetectionResult]:
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thr = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    closed = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, kernel, iterations=2)
    cnts, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    results = []
    for c in cnts:
        area = cv2.contourArea(c)
        if area < (h * w) * 0.01:
            continue
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) != 4:
            rect = cv2.minAreaRect(c)
            box = cv2.boxPoints(rect).astype(np.float32)
        else:
            box = approx.reshape(4, 2).astype(np.float32)
        d01 = np.linalg.norm(box[0] - box[1])
        d12 = np.linalg.norm(box[1] - box[2])
        aspect = max(d01, d12) / (min(d01, d12) + 1e-5)
        if aspect < 2.0:
            continue
        warped = _four_point_transform(image, box)
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [box.astype(np.int32)], 255)
        size = (warped.shape[1], warped.shape[0])
        results.append(CardDetectionResult(box=box, size=size, warped=warped, mask=mask))
    results.sort(key=lambda r: r.size[0] * r.size[1], reverse=True)
    return results

def _color_mask_hsv(hsv: np.ndarray, color: str) -> np.ndarray:
    if color == "red":
        m1 = cv2.inRange(hsv, (0, 80, 80), (10, 255, 255))
        m2 = cv2.inRange(hsv, (160, 80, 80), (180, 255, 255))
        return cv2.bitwise_or(m1, m2)
    if color == "yellow":
        return cv2.inRange(hsv, (20, 100, 100), (35, 255, 255))
    if color == "blue":
        return cv2.inRange(hsv, (100, 100, 80), (130, 255, 255))
    return np.zeros(hsv.shape[:2], dtype=np.uint8)

def detect_chips_in_card(card: CardDetectionResult) -> list[dict]:
    roi = card.warped
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    colors = ["yellow", "blue", "red"]
    chips = []
    for color in colors:
        mask = _color_mask_hsv(hsv, color)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            area = cv2.contourArea(c)
            if area < 20 or area > 6000:
                print(f"chip area: {area}")
                continue
            x, y, ww, hh = cv2.boundingRect(c)
            if color == "yellow" and hh > 40:
                print(f"yellow chip: {x}, {y}, {ww}, {hh}")
                continue

            cx = x + ww // 2
            cy = y + hh // 2
            chips.append({"center": (cx, cy), "bbox": (x, y, ww, hh), "color": color})
    return chips

def _is_landscape(size: tuple) -> bool:
    return size[0] > size[1]

def _group_chips_by_x(chips: list[dict], card_size: tuple) -> list[list[dict]]:
    if not chips:
        return []
    n = len(chips)
    widths = [c["bbox"][2] for c in chips]
    heights = [c["bbox"][3] for c in chips]
    med_w = float(np.median(widths)) if widths else 0.0
    med_h = float(np.median(heights)) if heights else 0.0
    chips_sorted = sorted(chips, key=lambda c: c["bbox"][0])
    x_positions = [c["bbox"][0] for c in chips_sorted]
    x_diffs = [abs(x_positions[i] - x_positions[i - 1]) for i in range(1, len(x_positions))]
    med_dx = float(np.median(x_diffs)) if x_diffs else med_w
    base = max(50.0, med_w * 1.0, med_dx * 0.8)
    colors = [c["color"] for c in chips]
    dominant_color = max(set(colors), key=colors.count) if colors else "blue"
    target_upper = 150.0 if dominant_color == "yellow" else 100.0
    
    # Calculate density based on card height and adjust for yellow boards
    card_h = float(card_size[0])
    effective_h = card_h
    if dominant_color == "yellow":
        effective_h = max(1.0, card_h - 100.0)
    
    # Density factor: arbitrary scaling to match previous logic range roughly
    # Previous: n / 20.0
    # New: proportional to n / effective_h
    # Assuming standard height ~500? n/20 ~= n * 25 / 500
    density = min((n * 50.0) / effective_h, 1.0)
    
    tolerance = base + density * (target_upper - base)
    tolerance = int(np.clip(tolerance, 50, 150))
    chips_sorted = sorted(chips, key=lambda c: c["bbox"][0])
    groups = []
    current = [chips_sorted[0]]
    base_x = chips_sorted[0]["bbox"][0]
    print("tolerance" , tolerance)
    for c in chips_sorted[1:]:
        x0 = c["bbox"][0]
        print(x0 , base_x , abs(x0 - base_x))
        if abs(x0 - base_x) <= tolerance:
            current.append(c)
        else:
            groups.append(current)
            current = [c]
            base_x = x0
    groups.append(current)
    return groups

def compute_unit_value(unit: list[dict]) -> int:

    return 100

def draw_overlay(image: np.ndarray, cards: list[CardDetectionResult], chips_per_card: list[list[dict]]) -> np.ndarray:
    out = image.copy()
    for idx, card in enumerate(cards):
        #print(f"card{idx}: {card.size}", card.box)
        box = card.box.astype(np.int32)
        cv2.polylines(out, [box], True, (0, 255, 0), 2)
        wpx, hpx = card.size
        label = f"{wpx}x{hpx}px"
        p = tuple(box[0].astype(int))
        rect = _order_points(card.box.astype(np.float32))
        cv2.putText(out, label, p, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        dst = np.array([[0, 0], [wpx - 1, 0], [wpx - 1, hpx - 1], [0, hpx - 1]], dtype=np.float32)
        Minv = cv2.getPerspectiveTransform(dst, rect)
        for chip in chips_per_card[idx]:
            x0, y0, ww, hh = chip["bbox"]
            corners = np.array([[x0, y0], [x0 + ww, y0], [x0 + ww, y0 + hh], [x0, y0 + hh]], dtype=np.float32)
            corners = corners.reshape(1, -1, 2)
            mapped = cv2.perspectiveTransform(corners, Minv).reshape(-1, 2).astype(np.int32)
            cv2.polylines(out, [mapped], True, (255, 0, 0), 2)
            pt_label = tuple(mapped[0])
            print(chip["color"] , x0 , y0 , ww , hh)
            cv2.putText(out, f"({ww}, {hh})", pt_label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    return out


def analyze_image(image: np.ndarray) -> dict:
    cards = detect_cards(image)
    chips_per_card = [detect_chips_in_card(c) for c in cards]
    overlay = draw_overlay(image, cards, chips_per_card)
    cards_info = []

    total_value = 0
    for c, chips in zip(cards, chips_per_card):
        units = _group_chips_by_x(chips , c.size) if _is_landscape(c.size) else []
        
        print(f" units={units}")
        
        unit_values = [compute_unit_value(u) for u in units]

        chip_distribution = [[classify_chip(c["bbox"][2]) for c in unit] for unit in units]
        
        print(chip_distribution)

        dominant_color = Counter([c["color"] for c in chips]).most_common(1)[0][0]

        print(dominant_color)
        
        figure = [figure_from_array(chip_dist) for chip_dist in chip_distribution]
    
        print(figure)

        

        if not figure:
            continue
        total = compute_output_board(dominant_color, figure)

        print(f"total={total}")

        total_value += total


        cards_info.append({"size_px": c.size, "box": c.box.tolist(), "chips": chips, "units": [{"chip_count": len(u), "value": v} for u, v in zip(units, unit_values)], "total_value": total})
    
    cv2.putText(overlay, f"Total Value: {total_value}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    return {"cards": cards_info, "overlay": overlay}

def analyze_image_with_params(image: np.ndarray, params: DetectParams) -> dict:
    cards, dbg = detect_cards_with_params(image, params)
    chips_per_card = [detect_chips_in_card(c) for c in cards]
    overlay = draw_overlay(image, cards, chips_per_card)
    cards_info = []
    for c, chips in zip(cards, chips_per_card):
        units = _group_chips_by_x(chips, c.size) if _is_landscape(c.size) else []
        unit_values = [compute_unit_value(u) for u in units]
        total_value = sum(unit_values)
        cards_info.append({"size_px": c.size, "box": c.box.tolist(), "chips": chips, "units": [{"chip_count": len(u), "value": v} for u, v in zip(units, unit_values)], "total_value": total_value})
    return {"cards": cards_info, "overlay": overlay, "debug": dbg}
