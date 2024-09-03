import torch
from ultralytics import YOLO

def get_device():
    """
    Determine the computing device to use for training.

    Checks if CUDA (GPU) is available. If it is, returns the index of the first CUDA device (0).
    If CUDA is not available, prints a message and returns 'cpu' to use the CPU instead.

    Returns:
        int or str: The device index for CUDA or 'cpu' if CUDA is not available.
    """
    if torch.cuda.is_available():
        return 0  # Use the first CUDA device
    else:
        print("CUDA is not available. Using CPU instead.")
        return "cpu"  # Use CPU
    
def train():
    """
    Train a classification model with the specified configuration.
    """
    model = YOLO("yolov8s-cls.yaml") 

    model.train(
        data="../clean_dataset",  # Path to training data
        imgsz=224,   # Image input size
        rect=False,  # Use rectangular training batches, must set to False for square input
        amp=True,  # Use rectangular training batches (False by default)
        epochs=100,  # Enable automatic mixed precision (fp16/fp32)
        patience=70,  # Patience for early stopping
        batch=128,  # Batch size
        mosaic=0,   # Mosaic augmentation
        close_mosaic=10,  # Close mosaic parameter, perform mosaic on last N epochs
        erasing=0.01,  # Erasing augmentation
        perspective=0,  # Perspective augmentation
        fliplr=0.1,  # Probability of horizontal flip
        flipud=0.1,  # Probability of vertical flip
        degrees=1,  # Range of degrees for random rotation
        scale=0.05,  # Range of scale augmentation
        translate=0.05,   # Range of translation augmentation
        optimizer="AdamW",   # Optimizer to use
        deterministic=True,   # Use deterministic training
        seed=42,   # Random seed for reproducibility
        device=get_device(), # Device to use (CPU or CUDA)
        cos_lr=False, # Use cosine learning rate scheduler
        lr0=0.00175, # Initial learning rate
        lrf=0.001, # Final learning rate scaling factor
        momentum=0.93,  # Momentum for the optimizer
        workers=8   # Number of data loading workers
    )


if __name__ == '__main__':
    train()
