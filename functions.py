import cv2
import numpy as np

## Image Sharpening Functions

def sharpen_image(image):
    # Gaussian blur the image
    blur = cv2.GaussianBlur(image, (0, 0), sigmaX=1.5)
    # Unsharp masking: original + weight * (original - blur)
    sharp = cv2.addWeighted(image, 1.5, blur, -0.5, 0)
    return sharp

## Image Enhancement Functions

def laplacian_sharpen(image):
    lap = cv2.Laplacian(image, cv2.CV_64F)
    lap = np.uint8(np.clip(lap, 0, 255))
    sharp = cv2.subtract(image, lap)
    return sharp

## Image Morphology Functions

from skimage.morphology import skeletonize
from skimage.util import invert

def thin_edges(binary_image):
    # Ensure binary (0 and 1)
    bin_img = binary_image // 255
    # Skeletonize
    skeleton = skeletonize(bin_img)
    return (skeleton * 255).astype(np.uint8)

## Calculate EPI (Edge Preservation Index)

def calculate_epi(ref_img, proc_img):
    # Convert both to float for accurate computation
    ref = ref_img.astype(np.float32)
    proc = proc_img.astype(np.float32)

    # Use Sobel operator to calculate gradient magnitudes
    grad_ref = cv2.Sobel(ref, cv2.CV_32F, 1, 0, ksize=3)**2 + cv2.Sobel(ref, cv2.CV_32F, 0, 1, ksize=3)**2
    grad_proc = cv2.Sobel(proc, cv2.CV_32F, 1, 0, ksize=3)**2 + cv2.Sobel(proc, cv2.CV_32F, 0, 1, ksize=3)**2

    # Square root to get magnitude
    grad_ref = np.sqrt(grad_ref)
    grad_proc = np.sqrt(grad_proc)

    # EPI: Edge preservation index = correlation between gradients
    numerator = np.sum((grad_ref - grad_ref.mean()) * (grad_proc - grad_proc.mean()))
    denominator = np.sqrt(np.sum((grad_ref - grad_ref.mean())**2) * np.sum((grad_proc - grad_proc.mean())**2))

    epi = numerator / denominator if denominator != 0 else 0
    return epi

def cal_mse(imageA, imageB):
    # Ensure both images have the same size
    err = np.mean((imageA.astype("float") - imageB.astype("float")) ** 2)
    return err

# --- Function to calculate PSNR ---
def psnr(imageA, imageB):
    mse_value = cal_mse(imageA, imageB)
    if mse_value == 0:
        return float('inf')  # Perfect match
    PIXEL_MAX = 255.0
    return 20 * np.log10(PIXEL_MAX / np.sqrt(mse_value))

# --- Function to calculate SSIM ---
def ssim_score(imageA, imageB):
    # Convert to grayscale if needed
    if len(imageA.shape) == 3:
        imageA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
    if len(imageB.shape) == 3:
        imageB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)
    score, _ = ssim(imageA, imageB, full=True)
    return score