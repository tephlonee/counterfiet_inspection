import argparse
import os
import glob
import time
import cv2
import numpy as np
from detection import analyze_image, DetectParams, analyze_image_with_params

def _iter_image_paths(input_path: str) -> list[str]:
    if os.path.isdir(input_path):
        exts = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff"]
        paths = []
        for e in exts:
            paths.extend(glob.glob(os.path.join(input_path, e)))
        paths.sort()
        return paths
    return [input_path]

def _ensure_dir(path: str | None) -> str | None:
    if path is None:
        return None
    os.makedirs(path, exist_ok=True)
    return path

def process_images(images_path: str, display: bool, save_dir: str | None):
    paths = _iter_image_paths(images_path)
    for p in paths[:1]:
        img = cv2.imread(p)
        if img is None:
            continue
        res = analyze_image(img)
        overlay = res["overlay"]
        cards = res["cards"]
        info = []
        for i, c in enumerate(cards):
            size = c["size_px"]
            chips = [(x["color"], x["center"]) for x in c["chips"]]
            info.append(f"card{i}: size={size}, chips={chips}")
        print(f"{os.path.basename(p)} -> " + "; ".join(info))
        if display:
            cv2.imshow("overlay", overlay)
            cv2.waitKey(1)
        if save_dir:
            name = os.path.splitext(os.path.basename(p))[0]
            out_path = os.path.join(save_dir, f"{name}_overlay.png")
            cv2.imwrite(out_path, overlay)
    if display:
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def process_video(source: str | int, display: bool, save_dir: str | None):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        return
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        res = analyze_image(frame)
        overlay = res["overlay"]
        cards = res["cards"]
        info = []
        for i, c in enumerate(cards):
            size = c["size_px"]
            chips = [(x["color"], x["center"]) for x in c["chips"]]
            info.append(f"card{i}: size={size}, chips={chips}")
        print(f"frame {idx} -> " + "; ".join(info))
        idx += 1
        if display:
            cv2.imshow("overlay", overlay)
            k = cv2.waitKey(1) & 0xFF
            if k == 27 or k == ord('q'):
                break
        if save_dir:
            ts = int(time.time() * 1000)
            out_path = os.path.join(save_dir, f"frame_{ts}_overlay.png")
            cv2.imwrite(out_path, overlay)
    cap.release()
    if display:
        cv2.destroyAllWindows()

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--images", type=str, default=None)
    p.add_argument("--video", type=str, default=None)
    p.add_argument("--webcam", type=int, default=None)
    p.add_argument("--display", action="store_true")
    p.add_argument("--save", type=str, default=None)
    p.add_argument("--tune", action="store_true")
    return p.parse_args()

def _create_trackbar_window():
    cv2.namedWindow("Tune", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Tune", 400, 500)
    cv2.createTrackbar("BlurK", "Tune", 5, 31, lambda x: None)
    cv2.createTrackbar("CannyL", "Tune", 50, 255, lambda x: None)
    cv2.createTrackbar("CannyU", "Tune", 150, 255, lambda x: None)
    cv2.createTrackbar("Smax", "Tune", 80, 255, lambda x: None)
    cv2.createTrackbar("Vmax", "Tune", 90, 255, lambda x: None)
    cv2.createTrackbar("MorphK", "Tune", 7, 25, lambda x: None)
    cv2.createTrackbar("MorphIter", "Tune", 2, 5, lambda x: None)
    cv2.createTrackbar("AreaMin%", "Tune", 1, 20, lambda x: None)
    cv2.createTrackbar("AspectMinx10", "Tune", 20, 50, lambda x: None)

def _read_params_from_trackbar() -> DetectParams:
    bk = cv2.getTrackbarPos("BlurK", "Tune")
    cl = cv2.getTrackbarPos("CannyL", "Tune")
    cu = cv2.getTrackbarPos("CannyU", "Tune")
    smax = cv2.getTrackbarPos("Smax", "Tune")
    vmax = cv2.getTrackbarPos("Vmax", "Tune")
    mk = cv2.getTrackbarPos("MorphK", "Tune")
    mi = cv2.getTrackbarPos("MorphIter", "Tune")
    area = cv2.getTrackbarPos("AreaMin%", "Tune")
    aspect10 = cv2.getTrackbarPos("AspectMinx10", "Tune")
    params = DetectParams(
        blur_ksize=bk if bk > 0 else 1,
        canny_lower=cl,
        canny_upper=cu,
        dark_s_max=smax,
        dark_v_max=vmax,
        morph_kernel=mk if mk > 1 else 1,
        morph_iter=mi,
        area_min_percent=float(area),
        aspect_min=max(1.0, aspect10 / 10.0),
    )
    return params

def tune_on_image(image: np.ndarray, save_dir: str | None):
    _create_trackbar_window()
    while True:
        params = _read_params_from_trackbar()
        res = analyze_image_with_params(image, params)
        overlay = res["overlay"]
        dbg = res["debug"]
        cv2.imshow("overlay", overlay)
        cv2.imshow("masked_edges", dbg["masked_edges"])
        cv2.imshow("mask_dark", dbg["mask_dark"])
        cv2.imshow("closed", dbg["closed"])
        cv2.imshow("blur", dbg["blur"])
        k = cv2.waitKey(30) & 0xFF
        if k == 27 or k == ord('q'):
            break
        if save_dir and k == ord('s'):
            out_path = os.path.join(save_dir, "tune_overlay.png")
            cv2.imwrite(out_path, overlay)
    cv2.destroyAllWindows()

def tune_on_video(source: str | int, save_dir: str | None):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        return
    _create_trackbar_window()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        params = _read_params_from_trackbar()
        res = analyze_image_with_params(frame, params)
        overlay = res["overlay"]
        dbg = res["debug"]
        cv2.imshow("overlay", overlay)
        cv2.imshow("masked_edges", dbg["masked_edges"])
        cv2.imshow("mask_dark", dbg["mask_dark"])
        cv2.imshow("closed", dbg["closed"])
        cv2.imshow("blur", dbg["blur"])
        k = cv2.waitKey(1) & 0xFF
        if k == 27 or k == ord('q'):
            break
        if save_dir and k == ord('s'):
            name = f"tune_overlay_{int(time.time()*1000)}.png"
            cv2.imwrite(os.path.join(save_dir, name), overlay)
    cap.release()
    cv2.destroyAllWindows()

def main():
    args = parse_args()
    save_dir = _ensure_dir(args.save)
    if args.tune:
        if args.images:
            paths = _iter_image_paths(args.images)
            if paths:
                img = cv2.imread(paths[0])
                if img is not None:
                    tune_on_image(img, save_dir)
                    return
        if args.video:
            tune_on_video(args.video, save_dir)
            return
        if args.webcam is not None:
            tune_on_video(args.webcam, save_dir)
            return
        default_dir = os.path.join(os.path.dirname(__file__), "captured_images")
        if os.path.isdir(default_dir):
            paths = _iter_image_paths(default_dir)
            if paths:
                img = cv2.imread(paths[0])
                if img is not None:
                    tune_on_image(img, save_dir)
                    return
    if args.images:
        process_images(args.images, args.display, save_dir)
        return
    if args.video:
        process_video(args.video, args.display, save_dir)
        return
    if args.webcam is not None:
        process_video(args.webcam, args.display, save_dir)
        return
    default_dir = os.path.join(os.path.dirname(__file__), "captured_images")
    if os.path.isdir(default_dir):
        process_images(default_dir, args.display, save_dir)
    else:
        print("No input provided.")

if __name__ == "__main__":
    main()

