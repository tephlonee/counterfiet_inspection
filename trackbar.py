
import cv2

from dataclasses import dataclass


@dataclass
class DetectParams:
    blur_ksize: int = 5
    canny_lower: int = 50
    canny_upper: int = 150
    thresh_val: int = 127
    dark_s_max: int = 80
    dark_v_max: int = 90
    morph_kernel: int = 7
    morph_iter: int = 2
    area_min_percent: float = 1.0
    aspect_min: float = 2.0

def create_trackbar():

    cv2.createTrackbar("Blur", "Processed", 3, 20, nothing)
    cv2.createTrackbar("Threshold", "Processed", 127, 255, nothing)
    #cv2.createTrackbar("Block Size", "Processed", 1, 20, nothing)
    cv2.createTrackbar("Canny Low", "Processed", 80, 255, nothing)
    cv2.createTrackbar("Canny High", "Processed", 200, 255, nothing)
    cv2.createTrackbar("Kernel", "Processed", 5, 20, nothing)
    cv2.createTrackbar("Morph Iter", "Processed", 3, 5, nothing)

def _read_params_from_trackbar() -> DetectParams:

    ksize = cv2.getTrackbarPos("Blur", "Processed") * 2 + 1  # kernel must be odd
    thresh_val = cv2.getTrackbarPos("Threshold", "Processed")
    canny_low = cv2.getTrackbarPos("Canny Low", "Processed")
    canny_high = cv2.getTrackbarPos("Canny High", "Processed")
    #block_size = cv2.getTrackbarPos("Block Size", "Processed") * 2 + 1
    kernel_size = cv2.getTrackbarPos("Kernel", "Processed") * 2 + 1
    morph_iter = cv2.getTrackbarPos("Morph Iter", "Processed")
    
    params = DetectParams(
        blur_ksize=ksize if ksize > 0 else 1,
        thresh_val=thresh_val,
        canny_lower=canny_low,
        canny_upper=canny_high,
        morph_kernel=kernel_size if kernel_size > 1 else 1,
        morph_iter=morph_iter,
    )
    return params