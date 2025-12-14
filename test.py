import cv2
import numpy as np

def show_resized(win_name, img, max_w=1280, max_h=720):
    h, w = img.shape[:2]
    scale = min(max_w / w, max_h / h, 1.0)
    if scale < 1.0:
        new_size = (int(w * scale), int(h * scale))
        interp = cv2.INTER_AREA if img.ndim in (2, 3) else cv2.INTER_NEAREST
        img = cv2.resize(img, new_size, interpolation=interp)
    cv2.imshow(win_name, img)

def detect_cards(image: np.ndarray):

    cv2.namedWindow("Tune", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Edges", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Closed", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Kernel", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Tune", 400, 500)
    cv2.createTrackbar("BlurK", "Tune", 5, 31, lambda x: None)
    cv2.createTrackbar("CannyL", "Tune", 50, 255, lambda x: None)
    cv2.createTrackbar("CannyU", "Tune", 150, 255, lambda x: None)
    #cv2.createTrackbar("Smax", "Tune", 80, 255, lambda x: None)
   # cv2.createTrackbar("Vmax", "Tune", 90, 255, lambda x: None)
    cv2.createTrackbar("MorphK", "Tune", 7, 25, lambda x: None)
    cv2.createTrackbar("MorphIter", "Tune", 2, 5, lambda x: None)
    cv2.createTrackbar("AreaMin%", "Tune", 1, 20, lambda x: None)
    #cv2.createTrackbar("AspectMinx10", "Tune", 20, 50, lambda x: None)

    while True:

        bk = cv2.getTrackbarPos("BlurK", "Tune")
        cl = cv2.getTrackbarPos("CannyL", "Tune")
        cu = cv2.getTrackbarPos("CannyU", "Tune")
        #smax = cv2.getTrackbarPos("Smax", "Tune")
        #vmax = cv2.getTrackbarPos("Vmax", "Tune")
        mk = cv2.getTrackbarPos("MorphK", "Tune")
        mi = cv2.getTrackbarPos("MorphIter", "Tune")
        area = cv2.getTrackbarPos("AreaMin%", "Tune")
        #aspect10 = cv2.getTrackbarPos("AspectMinx10", "Tune")

        bk = max(1, bk)
        if bk % 2 == 0:
            bk += 1

        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (bk, bk), 0)
        _, thr = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        edges = cv2.Canny(thr, cl, cu)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (mk, mk))
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=mi)
        cnts, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        results = []
        for c in cnts:

            try:
                area = cv2.contourArea(c)
                if area < h * w * area / 100.0:
                    continue
                aspect = cv2.minAreaRect(c)[1][0] / cv2.minAreaRect(c)[1][1]
                if aspect < aspect_min:
                    continue
                results.append(c)
            except:
                continue

            rect = cv2.minAreaRect(c)
            box = cv2.boxPoints(rect)
            box = np.int8(box)
            cv2.drawContours(image, [box], 0, (0, 255, 0), 2)

        cv2.imshow("Tuned", closed)
        k = cv2.waitKey(30) & 0xFF
        if k == 27 or k == ord('q'):
            break





def read_image(image_path: str) -> np.ndarray:
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Failed to read image: {image_path}")
    return img


def main():
    image_path = "captured_images/capture_20251211_143050.jpg"
    img = read_image(image_path)
    detect_cards(img)

main()