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


def plot_confusion_matrix(cm, saveToFile=None, annot=True, fmt="d", cmap="Blues", xticklabels=None, yticklabels=None):
    """
    Plots a heatmap of the confusion matrix.

    Parameters:
        cm (list of lists): The confusion matrix.
        annot (bool): Whether to annotate the heatmap with the cell values. Default is True.
        fmt (str): The format specifier for cell value annotations. Default is "d" (integer).
        cmap (str): The colormap for the heatmap. Default is "Blues".
        xticklabels (list): Labels for the x-axis ticks. Default is None.
        yticklabels (list): Labels for the y-axis ticks. Default is None.

    Returns:
        None
    """
    
    # Convert the confusion matrix to a NumPy array
    cm = np.array(cm)

    # Create a figure and axis for the heatmap
    fig, ax = plt.subplots()

    # Plot the heatmap
    im = ax.imshow(cm, cmap=cmap)
    
    # Display cell values as annotations
    if annot:
        # Normalize the colormap to get values between 0 and 1
        norm = Normalize(vmin=cm.min(), vmax=cm.max())
        for i in range(len(cm)):
            for j in range(len(cm[i])):
                value = cm[i, j]
                # Determine text color based on cell value
                text_color = 'white' if norm(value) > 0.5 else 'black'  
                text = ax.text(j, i, format(value, fmt), ha="center", va="center", color=text_color)

    # Set x-axis and y-axis ticks and labels
    if xticklabels:
        ax.set_xticks(np.arange(len(xticklabels)))
        ax.set_xticklabels(xticklabels)
    if yticklabels:
        ax.set_yticks(np.arange(len(yticklabels)))
        ax.set_yticklabels(yticklabels)

    # Set labels and title
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix Heatmap")

    # Add a colorbar
    cbar = ax.figure.colorbar(im, ax=ax)

    # Show the plot
    if(saveToFile is not None):
        plt.savefig(saveToFile)
        
    plt.show()