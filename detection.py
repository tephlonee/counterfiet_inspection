import cv2
import numpy as np
from dataclasses import dataclass

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
    mask_dark = cv2.inRange(hsv, (0, 0, 0), (180, 80, 90))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    closed = cv2.morphologyEx(mask_dark, cv2.MORPH_CLOSE, kernel, iterations=2)
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
            if area < 20 or area > 5000:
                continue
            x, y, ww, hh = cv2.boundingRect(c)
            cx = x + ww // 2
            cy = y + hh // 2
            chips.append({"center": (cx, cy), "bbox": (x, y, ww, hh), "color": color})
    return chips

def draw_overlay(image: np.ndarray, cards: list[CardDetectionResult], chips_per_card: list[list[dict]]) -> np.ndarray:
    out = image.copy()
    for idx, card in enumerate(cards):
        box = card.box.astype(np.int32)
        cv2.polylines(out, [box], True, (0, 255, 0), 2)
        wpx, hpx = card.size
        label = f"{wpx}x{hpx}px"
        p = tuple(box[0].astype(int))
        cv2.putText(out, label, p, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        for chip in chips_per_card[idx]:
            x, y = chip["center"]
            x0, y0, ww, hh = chip["bbox"]
            cv2.rectangle(out, (x0, y0), (x0 + ww, y0 + hh), (255, 0, 0), 2)
            cv2.putText(out, chip["color"], (x0, max(0, y0 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    return out

def analyze_image(image: np.ndarray) -> dict:
    cards = detect_cards_binary(image)
    if not cards:
        cards = detect_cards(image)
    chips_per_card = [detect_chips_in_card(c) for c in cards]
    overlay = draw_overlay(image, cards, chips_per_card)
    cards_info = []
    for c, chips in zip(cards, chips_per_card):
        cards_info.append({"size_px": c.size, "box": c.box.tolist(), "chips": chips})
    return {"cards": cards_info, "overlay": overlay}

def analyze_image_with_params(image: np.ndarray, params: DetectParams) -> dict:
    cards, dbg = detect_cards_with_params(image, params)
    chips_per_card = [detect_chips_in_card(c) for c in cards]
    overlay = draw_overlay(image, cards, chips_per_card)
    cards_info = []
    for c, chips in zip(cards, chips_per_card):
        cards_info.append({"size_px": c.size, "box": c.box.tolist(), "chips": chips})
    return {"cards": cards_info, "overlay": overlay, "debug": dbg}
