import cv2
import numpy as np

def zoom_in(image, zoom_factor=0.1):
    # Calculate the new size after zooming
    h, w = image.shape[:2]
    new_w, new_h = int(w * (1 - zoom_factor)), int(h * (1 - zoom_factor))

    # Calculate the coordinates to crop the image to simulate zooming
    x1 = (w - new_w) // 2
    y1 = (h - new_h) // 2
    x2 = x1 + new_w
    y2 = y1 + new_h

    # Crop and resize the image to original size
    zoomed_image = image[y1:y2, x1:x2]
    zoomed_image = cv2.resize(zoomed_image, (w, h), interpolation=cv2.INTER_LINEAR)
    return zoomed_image

def gamma_correction(image):
    # Calculate the mean brightness of the image
    mean_brightness = np.mean(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY))
    
    # Automatic gamma correction
    gamma = np.log10(0.5) / np.log10(mean_brightness / 255.0)
    corrected_image = np.array(255 * (image / 255) ** gamma, dtype='uint8')
    return corrected_image

def histogram_stretch(image):
    if len(image.shape) == 2:  # Grayscale image
        # Normalize the grayscale image
        stretched_image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
    elif len(image.shape) == 3:  # Color image
        channels = cv2.split(image)
        stretched_channels = []
        for channel in channels:
            stretched_channel = cv2.normalize(channel, None, 0, 255, cv2.NORM_MINMAX)
            stretched_channels.append(stretched_channel)
        stretched_image = cv2.merge(stretched_channels)
    else:
        raise ValueError("Unsupported image format")

    return stretched_image

def apply_clahe(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    # Create a CLAHE object with specified clip limit and tile grid size
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    
    # Check if the image is grayscale (single-channel) or color (multi-channel)
    if len(image.shape) == 2:  # Grayscale image
        # Apply CLAHE to the grayscale image
        clahe_image = clahe.apply(image)
    elif len(image.shape) == 3:  # Color image
        # Split the color image into its respective channels
        channels = cv2.split(image)
        # Apply CLAHE to each channel separately
        clahe_channels = [clahe.apply(channel) for channel in channels]
        # Merge the CLAHE-enhanced channels back into a color image
        clahe_image = cv2.merge(clahe_channels)
    else:
        raise ValueError("Unsupported image format")
    
    return clahe_image
