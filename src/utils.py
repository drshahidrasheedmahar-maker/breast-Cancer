import cv2
import numpy as np


def fun_remove_bg(image):
    # Create a mask by thresholding (background is dark)
    _, mask = cv2.threshold(image, 30, 255, cv2.THRESH_BINARY)

    # Clean up mask with morphology
    kernel = np.ones((3,3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel, iterations=1)
    # Apply mask to keep only foreground
    bg_removed = cv2.bitwise_and(image, image, mask=mask)
    return bg_removed, mask