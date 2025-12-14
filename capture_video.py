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





# conecting to the first available camera
camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())

# Grabing Continusely (video) with minimal delay
camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly) 
converter = pylon.ImageFormatConverter()

# converting to opencv bgr format
converter.OutputPixelFormat = pylon.PixelType_BGR8packed
converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

cv2.namedWindow('basler live feed', cv2.WINDOW_NORMAL)
videos_dir = os.path.join(os.path.dirname(__file__), "captured_videos")
os.makedirs(videos_dir, exist_ok=True)
writer = None
recording = True
fps = 30.0
current_path = None

while camera.IsGrabbing():
    grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)

    if grabResult.GrabSucceeded():
        # Access the image data
        image = converter.Convert(grabResult)
        img = image.GetArray()
        gray_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Display the resulting frame
        cv2.imshow('basler live feed', img)
        if recording:
            if writer is None:
                ts = time.strftime("%Y%m%d_%H%M%S")
                current_path = os.path.join(videos_dir, f"record_{ts}.mp4")
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                h, w = img.shape[:2]
                writer = cv2.VideoWriter(current_path, fourcc, fps, (w, h))
            writer.write(img)
        
        #cv2.imshow('title', combined_img)
        k = cv2.waitKey(1)
        if k == 27:
            break
        if k == ord('r'):
            recording = not recording
            if not recording and writer is not None:
                writer.release()
                writer = None
                current_path = None
        if k == ord('n'):
            if writer is not None:
                writer.release()
                writer = None
            recording = True
    grabResult.Release()
    
# Releasing the resource    
camera.StopGrabbing()
if writer is not None:
    writer.release()

cv2.destroyAllWindows()
