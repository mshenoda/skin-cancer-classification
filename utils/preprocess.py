import os
import math
from concurrent.futures import ProcessPoolExecutor, as_completed
import cv2
import numpy as np
from tqdm import tqdm

def evaluate_mask(mask, original_gray) -> float:
    """
    Evaluates the quality of a binary mask by considering its coverage, structure,
    and difference from the original grayscale image.
    
    Parameters:
    - mask: The binary mask to be evaluated.
    - original_gray: The original grayscale image.
    
    Returns:
    - score: A score indicating the quality of the mask (lower is better).
    """
    # Calculate the coverage of the mask (how much of the image is being masked)
    mask_coverage = np.sum(mask) / (255.0 * mask.size)  # Normalized coverage

    # Consider the structure/continuity of the mask using edges
    edges = cv2.Canny(mask, 15, 150)
    edge_density = np.sum(edges) / 255.0  # Normalized edge density

    # Calculate the difference between the original grayscale image and the masked regions
    masked_gray = cv2.bitwise_and(original_gray, original_gray, mask=mask)
    difference = np.sum(cv2.absdiff(original_gray, masked_gray)) / (255.0 * original_gray.size)  # Normalized difference

    # Combine the coverage, edge density, and difference for a final score
    # Adjust the weights of each component if needed
    score = (0.5 * mask_coverage) + (0.3 * edge_density) + (0.2 * difference)
    
    return score

def clean_mask(mask, min_area=20):
    """
    Cleans a binary mask by removing noisy dots and keeping continuous curvy lines.
    
    Parameters:
    - mask: The binary mask to be cleaned (2D numpy array).
    - min_area: Minimum area of connected components to keep (default: 1000 pixels).
    
    Returns:
    - cleaned_mask: The cleaned binary mask (2D numpy array).
    """
    # Convert mask to uint8 type if it is not already
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)

    # Define kernel for morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 1))

    # Apply morphological opening to remove small noise
    opened_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    
    # Apply morphological closing to fill small holes
    closed_mask = cv2.morphologyEx(opened_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Find connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(closed_mask, connectivity=8)

    # Create a new mask to keep only the large components
    cleaned_mask = np.zeros_like(mask, dtype=np.uint8)

    for i in range(1, num_labels):  # Start from 1 to skip the background component
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            cleaned_mask[labels == i] = 255

    return cleaned_mask

def is_mask_almost_empty(mask, threshold=0.3) -> bool:
    """
    Checks if the percentage of non-zero pixels in a binary mask is below a given threshold.
    
    Parameters:
    - mask: The binary mask to be evaluated (2D numpy array).
    - threshold: The percentage threshold below which the mask is considered "almost empty" (default: 1.0).
    
    Returns:
    - bool: True if the percentage of non-zero pixels is below the threshold, False otherwise.
    """
    # Calculate the total number of pixels in the mask
    total_pixels = mask.size

    # Calculate the percentage of non-zero pixels
    non_zero_count = np.count_nonzero(mask)
    non_zero_percentage = (non_zero_count / total_pixels) * 100

    # Return True if the non-zero percentage is below the threshold, otherwise False
    return non_zero_percentage < threshold


def is_mask_almost_full_at_center(mask, center_fraction=0.5, threshold=60.0) -> bool:
    """
    Checks if the percentage of non-zero pixels in the center region of a binary mask is above a given threshold.
    
    Parameters:
    - mask: The binary mask to be evaluated (2D numpy array).
    - center_fraction: The fraction of the mask's dimensions to consider as the center region (default: 0.5, i.e., the central 50%).
    - threshold: The percentage threshold above which the center region is considered "almost full" (default: 60.0).
    
    Returns:
    - bool: True if the percentage of non-zero pixels in the center region is above the threshold, False otherwise.
    """
    # Get the dimensions of the mask
    height, width = mask.shape

    # Calculate the center region's boundaries
    center_h_start = int((1 - center_fraction) * height / 2)
    center_h_end = int((1 + center_fraction) * height / 2)
    center_w_start = int((1 - center_fraction) * width / 2)
    center_w_end = int((1 + center_fraction) * width / 2)

    # Extract the center region from the mask
    center_region = mask[center_h_start:center_h_end, center_w_start:center_w_end]

    # Calculate the total number of pixels in the center region
    total_center_pixels = center_region.size

    # Calculate the percentage of non-zero pixels in the center region
    non_zero_center_count = np.count_nonzero(center_region)
    non_zero_center_percentage = (non_zero_center_count / total_center_pixels) * 100

    # Return True if the non-zero percentage in the center is above the threshold, otherwise False
    return non_zero_center_percentage > threshold

def remove_thin_lines(image):
    """
    Removes hairlines from a skin cancer image while preserving the original content of the skin.
    Only removes hair if hair is detected in the image, also removes other thinlines found in image.
    
    Parameters:
    - image: Input image in which hair needs to be removed (OpenCV format).
    
    Returns:
    - inpainted_image: Resulting image with hair removed (OpenCV format).
    - best_mask: The binary mask used for hair removal (2D numpy array).
    """

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    best_mask = None
    best_score = float('inf')

    # Adaptive kernel size range
    min_kernel_size = 7
    max_kernel_size = 15
    kernel_step = 2
    erode_dilate_iterations=1
    
    # Iterate over different kernel sizes
    for kernel_size in range(min_kernel_size, max_kernel_size + 1, kernel_step):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        
        # Apply blackhat morphology to detect hair
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)

        # Thresholding range
        min_thresh = 7 
        max_thresh = 15
        step = 1

        for thresh in range(min_thresh, max_thresh + 1, step):
            _, binary_mask = cv2.threshold(blackhat, thresh, 255, cv2.THRESH_BINARY)

            if is_mask_almost_full_at_center(binary_mask):
                erode_dilate_iterations = 5

            small_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 1))
            
            binary_mask = cv2.erode(binary_mask, small_kernel, iterations=erode_dilate_iterations)
            binary_mask = cv2.dilate(binary_mask, small_kernel, iterations=erode_dilate_iterations)

            # Evaluate the mask quality
            score = evaluate_mask(binary_mask, gray)

            if score < best_score:
                best_score = score
                best_mask = binary_mask

    # Use the best mask found for inpainting
    cleaned_mask = clean_mask(best_mask)
    if is_mask_almost_empty(cleaned_mask):
        inpainted_image = image
    else:
        inpainted_image = cv2.inpaint(image, cleaned_mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)

    return inpainted_image, cleaned_mask


def detect_circle(image):
    """
    Detect the circular content in the image using HoughCircles.
    
    Parameters:
        image (np.ndarray): The input image.
    
    Returns:
        tuple: (center, radius) of the detected circle.
    """
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise 
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)
    
    # Detect circles using HoughCircles
    circles = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=20,
        param1=50, param2=30, minRadius=50, maxRadius=200
    )
    
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        # Assuming the first detected circle is the one we need
        return (circles[0][0], circles[0][1]), circles[0][2]
    else:
        return None, None

def crop_to_circle(image, center, radius):
    """
    Crop the image so that the circle touches the edges of the frame.
    
    Parameters:
        image (np.ndarray): The input image.
        center (tuple): The center (x, y) of the circle.
        radius (int): The radius of the circle.
    
    Returns:
        np.ndarray: The cropped image.
    """
    height, width, _ = image.shape
    x, y = center

    # Define the bounding box of the circle
    x1 = max(0, x - radius)
    x2 = min(width, x + radius)
    y1 = max(0, y - radius)
    y2 = min(height, y + radius)

    cropped_image = image[y1:y2, x1:x2]
    
    return cropped_image


def create_circle_mask(image, center, radius):
    """
    Create a mask for inpainting by detecting areas outside the circle.
    
    Parameters:
        image (np.ndarray): The input image.
        center (tuple): The center (x, y) of the circle.
        radius (int): The radius of the circle.
    
    Returns:
        np.ndarray: Mask where areas outside the circle are marked.
    """
    height, width = image.shape[:2]

    # Reduce the radius
    new_radius = radius - int(math.ceil(0.1 * radius))
    
    # Ensure the center is within the bounds after reducing the radius
    new_center_x = min(max(center[0], new_radius), width - new_radius)
    new_center_y = min(max(center[1], new_radius), height - new_radius)
    new_center = (new_center_x, new_center_y)

    # Create a white mask
    mask = np.ones((height, width), dtype=np.uint8) * 255
    
    # Draw a filled circle on the mask
    cv2.circle(mask, new_center, new_radius, (0, 0, 0), thickness=-1)
    
    return mask

def inpaint_black_areas(image, mask):
    """
    Inpaint the black areas in the image using the provided mask.
    
    Parameters:
        image (np.ndarray): The input image.
        mask (np.ndarray): Mask where areas to be inpainted are marked.
    
    Returns:
        np.ndarray: The inpainted image.
    """
    # Ensure mask is in the correct format (0 or 255)
    if len(mask.shape) == 2:
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)

    # Inpainting using OpenCV's inpaint function
    inpainted_image = cv2.inpaint(image, mask[:, :, 0], inpaintRadius=7, flags=cv2.INPAINT_TELEA)
    
    return inpainted_image

def create_blob_mask(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Step 1: Create initial binary mask using thresholding
    _, binary_mask = cv2.threshold(gray, 22, 255, cv2.THRESH_BINARY_INV)
    
    # Ensure binary_mask is a single-channel image of type uint8
    if len(binary_mask.shape) != 2:
        raise ValueError("The binary_mask must be a single-channel image.")
    if binary_mask.dtype != np.uint8:
        binary_mask = binary_mask.astype(np.uint8)
    
    # Create a mask for the central region
    h, w = binary_mask.shape
    center_x, center_y = w // 2, h // 2
    radius_x, radius_y = int(w * 0.38), int(h * 0.38)  # Adjusted to cover 70% of the image

    central_mask = np.zeros_like(binary_mask)
    cv2.ellipse(central_mask, (center_x, center_y), (radius_x, radius_y), 0, 0, 360, 255, -1)

    # Find blobs in the binary mask
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)

    # Create a mask to exclude blobs in the central region
    final_mask = binary_mask.copy()

    for i in range(1, num_labels):  # Skip label 0 (background)
        x, y = centroids[i]
        x, y = int(x), int(y)
        
        # Check if the blob's centroid is in the central region
        if central_mask[y, x] == 255:
            final_mask[labels == i] = 0

    # Apply erosion to the entire mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    # Apply dilation to the entire mask
    final_mask = cv2.dilate(final_mask, kernel, iterations=5)

    return final_mask

def clean_image_borders(image):
    """
    Process the image to detect the circular content, crop it, and remove black corners.
    
    Parameters:
        image_path (str): Path to the input image.
        output_path (str): Path to save the processed image.
    """
    
    mask = create_blob_mask(image)
    if not is_mask_almost_empty(mask):
        mask = cv2.GaussianBlur(mask, (15,15), 25)
        result_image = inpaint_black_areas(image, mask)
        return result_image, mask
    else:
        height, width, _ = image.shape
        # Detect the circular content
        _, binary_mask = cv2.threshold(image, 30, 255, cv2.THRESH_BINARY_INV)

        binary_mask = cv2.GaussianBlur(binary_mask, (15,15), 5)

        center, radius = detect_circle(binary_mask)

        if center is None:
            result_image = image
            height, width = image.shape[:2]
            mask = np.zeros((height, width), dtype=np.uint8)
            return result_image, mask

        cropped_image = crop_to_circle(image, center, radius)

        # Create mask for inpainting
        mask = create_circle_mask(cropped_image, (center[0] - (center[0] - radius), center[1] - (center[1] - radius)), radius)
        mask = cv2.GaussianBlur(mask, (15,15), 25)
        # Inpaint the black areas
        result_image = inpaint_black_areas(cropped_image, mask)
        result_image = cv2.resize(result_image, (width, height))
        return result_image, mask
    
def clean_image(image):
    result_image, hairline_mask = remove_thin_lines(image)  # Apply hair removal
    result_image, border_mask = clean_image_borders(result_image)  # Apply border cleaning
    return result_image, hairline_mask, border_mask  
    
class ImageCleaner:
    """
    A class for cleaning and processing images by removing thin lines and cleaning borders.

    Example:
        cleaner = ImageCleaner()
        cleaner.process_image('path/to/input/image.jpg', 'path/to/output/image.jpg')
        cleaner.process_directory('path/to/input/directory', 'path/to/output/directory')
    """

    def __init__(self, num_workers=os.cpu_count()):
        """
        Initializes the ImageCleaner with the specified number of worker processes.

        Args:
            num_workers (int): The number of worker processes to use for parallel processing. Defaults to the number of CPU cores.
        """
        self.num_workers = num_workers

    def clean_image(self, image):
        """
        Cleans the given image by removing thin lines and cleaning borders if necessary.

        Args:
            image (numpy.ndarray): The image to be cleaned.

        Returns:
            numpy.ndarray: The cleaned image.
        """
        return clean_image(image)

    def process_image(self, image_path, output_path):
        """
        Processes a single image by reading it from the specified path, cleaning it, and saving the result to the output path.

        Args:
            image_path (str): The path to the input image.
            output_path (str): The path where the cleaned image will be saved.
        """
        image = cv2.imread(image_path)
        
        if image is None:
            return

        result_image = self.clean_image(image)

        base_name = os.path.basename(image_path)
        result_image_path = os.path.join(output_path, base_name)
        cv2.imwrite(result_image_path, result_image)

    def process_directory(self, input_dir, output_dir):
        """
        Processes all images in the input directory, applying cleaning operations and saving the results to the output directory.
        The directory structure is preserved in the output directory.

        Args:
            input_dir (str): The path to the input directory containing images.
            output_dir (str): The path to the output directory where cleaned images will be saved.
        """
        image_paths = []
        output_dirs = set()
        
        # Walk through the directory to gather image paths and create output directories upfront
        for root, _, files in os.walk(input_dir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                    input_image_path = os.path.join(root, file)
                    image_paths.append(input_image_path)
                    
                    # Determine the output directory for the current image
                    relative_path = os.path.relpath(root, input_dir)
                    output_image_dir = os.path.join(output_dir, relative_path)
                    
                    if output_image_dir not in output_dirs:
                        output_dirs.add(output_image_dir)
                        if not os.path.exists(output_image_dir):
                            os.makedirs(output_image_dir)

        # Use ProcessPoolExecutor for parallel processing
        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            futures = []
            for image_path in image_paths:
                root = os.path.dirname(image_path)
                relative_path = os.path.relpath(root, input_dir)
                output_image_dir = os.path.join(output_dir, relative_path)
                
                # Submit the image processing task to the pool
                futures.append(executor.submit(self.process_image, image_path, output_image_dir))
            
            # Use tqdm to track progress of all submitted tasks
            for _ in tqdm(as_completed(futures), total=len(futures), desc="Processing images"):
                pass