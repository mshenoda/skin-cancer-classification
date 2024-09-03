import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from utils.files import filter_duplicates

def plot_images(images, titles=None, figsize=(12, 6)):
    """
    Plots a list of images side by side with optional titles.

    Args:
    - images (list of ndarray): List of images to be displayed.
    - titles (list of str, optional): List of titles for the images.
    - figsize (tuple, optional): Size of the figure, default is (12, 6).

    Example:
    plot_images([image, image_clean], ['Original Image', 'Cleaned Image'])
    """
    
    num_images = len(images)
    
    # Create a subplot with 1 row and num_images columns
    fig, ax = plt.subplots(1, num_images, figsize=figsize)

    if num_images == 1:
        ax = [ax]  # To ensure ax is always iterable

    for i in range(num_images):
        if len(images[i].shape) == 3:
            image_rgb = cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB)
            ax[i].imshow(image_rgb)
        else:
            ax[i].imshow(images[i])
        if titles:
            ax[i].set_title(titles[i])
        ax[i].axis('off')  # Hide axis

    plt.show()


def plot_duplicates(duplicates_dict, score_threshold=1.0):
    high_scored_duplicates = filter_duplicates(duplicates_dict, score_threshold)
    for original, duplicates in high_scored_duplicates.items():
        # Load and process the original image
        original_img = cv2.imread(original)

        # Prepare the list of images and titles
        images = [original_img]
        titles = [f'Original:\n{original.split("/")[-1]}']
        
        for duplicate, _ in duplicates:
            duplicate_img = cv2.imread(duplicate)
            images.append(duplicate_img)
            titles.append(f'Duplicate:\n{duplicate.split("/")[-1]}')
        
        # Plot all images using the updated plot_images function
        plot_images(images, titles)