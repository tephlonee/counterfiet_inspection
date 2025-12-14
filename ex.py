import cv2
import numpy as np

# Callback function for trackbars (does nothing)
def nothing(x):
    pass

# Load image

def detect_cards(image: np.ndarray):

    # Create a resizable window
    cv2.namedWindow("Processed" , cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Processed", 900, 500)

    cv2.namedWindow("Contour" , cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Contour", 900, 500)
    
    

    # Create trackbars
    cv2.createTrackbar("Blur", "Processed", 1, 20, nothing)
    cv2.createTrackbar("Threshold", "Processed", 0, 255, nothing)
    cv2.createTrackbar("Block Size", "Processed", 1, 20, nothing)
    cv2.createTrackbar("Canny Low", "Processed", 80, 255, nothing)
    cv2.createTrackbar("Canny High", "Processed", 200, 255, nothing)
    cv2.createTrackbar("Kernel", "Processed", 1, 20, nothing)
    cv2.createTrackbar("Morph Iter", "Processed", 1, 5, nothing)

    while True:
        # Get current trackbar positions
        ksize = cv2.getTrackbarPos("Blur", "Processed") * 2 + 1  # kernel must be odd
        thresh_val = cv2.getTrackbarPos("Threshold", "Processed")
        canny_low = cv2.getTrackbarPos("Canny Low", "Processed")
        canny_high = cv2.getTrackbarPos("Canny High", "Processed")
        block_size = cv2.getTrackbarPos("Block Size", "Processed") * 2 + 1
        kernel_size = cv2.getTrackbarPos("Kernel", "Processed") * 2 + 1
        morph_iter = cv2.getTrackbarPos("Morph Iter", "Processed")


        # Apply processing
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (ksize, ksize), 0)
        _, thresh = cv2.threshold(blurred, thresh_val, 255, cv2.THRESH_BINARY)
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel , iterations=morph_iter)
        blurred = cv2.GaussianBlur(closed, (ksize, ksize), 0)
        inverted = cv2.bitwise_not(blurred)

        edges = cv2.Canny(inverted, canny_low, canny_high)

        # Stack for visualization
        combined = np.hstack([blurred, thresh, closed, edges])
        
        cv2.imshow("Processed", combined),

        # Find contours on the edge image
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Draw contours on a copy of the original image for visualization


        if contours:

            
            contour_img = image.copy()
            cv2.drawContours(contour_img, contours, -1, (0, 255, 0), 1)
            # Stack the contour image with the previous combined view
            
            cv2.imshow("Contour", contour_img)

        combined = np.hstack([blurred, thresh, closed ,  inverted , edges])

        cv2.imshow("Processed", combined)
        
        # Break loop on ESC
        if cv2.waitKey(50) & 0xFF == 27:
            break

    cv2.destroyAllWindows()

def read_image(image_path: str) -> np.ndarray:
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Failed to read image: {image_path}")
    return img


def main():
    image_path = "captured_images/capture_20251211_143055.jpg"
    img = read_image(image_path)
    detect_cards(img)

main()
