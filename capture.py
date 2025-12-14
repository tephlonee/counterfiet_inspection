'''
A simple Program for grabing video from basler camera and converting it to opencv img.
Tested on Basler acA1300-200uc (USB3, linux 64bit , python 3.5)
https://www.geeksforgeeks.org/clahe-histogram-eqalization-opencv/
'''
from pypylon import pylon
import cv2
import numpy as np
import os
import time

def resize_for_window(img: np.ndarray, max_w: int = 1280, max_h: int = 720) -> np.ndarray:
    h, w = img.shape[:2]
    scale = min(max_w / float(w), max_h / float(h), 1.0)
    if scale < 1.0:
        return cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return img

def save_frame(save_dir: str, img: np.ndarray) -> str:
    ts = time.strftime("%Y%m%d_%H%M%S")
    path = os.path.join(save_dir, f"capture_{ts}.jpg")
    cv2.imwrite(path, img)
    return path

def auto_save_tick(enabled: bool, last_save: float, save_dir: str, img: np.ndarray, interval_s: float = 5.0) -> float:
    if not enabled:
        return last_save
    now = time.time()
    if (now - last_save) >= interval_s:
        save_frame(save_dir, img)
        return now
    return last_save




# conecting to the first available camera
camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())

# Grabing Continusely (video) with minimal delay
camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly) 
converter = pylon.ImageFormatConverter()

# converting to opencv bgr format
converter.OutputPixelFormat = pylon.PixelType_BGR8packed
converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

cv2.namedWindow('basler live feed', cv2.WINDOW_NORMAL)
save_dir = os.path.join(os.path.dirname(__file__), "captured_images")
os.makedirs(save_dir, exist_ok=True)
auto_capture = False
last_save = 0.0
interval_s = 5.0

while camera.IsGrabbing():
    grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)

    if grabResult.GrabSucceeded():
        # Access the image data
        image = converter.Convert(grabResult)
        img = image.GetArray()
        gray_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Display the resulting frame
        resized = resize_for_window(img, 1280, 720)
        cv2.imshow('basler live feed', resized)
        
        #cv2.imshow('title', combined_img)
        k = cv2.waitKey(1)
        if k == 27:
            break
        if k == ord('t'):
            auto_capture = not auto_capture
            last_save = time.time()
        if k == ord('s'):
            save_frame(save_dir, img)
        last_save = auto_save_tick(auto_capture, last_save, save_dir, img, interval_s)
    grabResult.Release()
    
# Releasing the resource    
camera.StopGrabbing()

cv2.destroyAllWindows()
