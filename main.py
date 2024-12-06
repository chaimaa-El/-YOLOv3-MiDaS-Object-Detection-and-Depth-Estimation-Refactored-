import subprocess
import sys

# Function to train the YOLOv3 model
def run_train():
    """
    This function trains the YOLOv3 model using the custom dataset.
    Arguments:
    - cfg: Configuration file for the YOLOv3 model.
    - data: Dataset configuration file containing information about the data.
    - epochs: Number of epochs to train the model.
    - batch_size: Batch size for training.
    - weights: Path to the initial weights file.
    - device: Device to use for training, e.g., GPU or CPU.
    """
    command = [
        'python', 'src/train.py',  # Running the training script
        '--cfg', 'src/cfg/mde.cfg',  # Config for YOLOv3
        '--data', 'data/customdata/custom.data',  # Custom dataset configuration
        '--epochs', '300',  # Number of epochs
        '--batch-size', '16',  # Training batch size
        '--weights', 'weights/best.pt',  # Weights file path
        '--device', '0'  # Use GPU 0
    ]
    subprocess.run(command)  # Execute the training command
    print(f"Training completed. ")

# Function to run object detection and depth estimation
def run_detect():
    """
    This function performs object detection and depth estimation on input images.
    Arguments:
    - cfg: Configuration file for YOLOv3.
    - names: Class names for detection.
    - weights: Weights file to use for detection.
    - source: Source for detection (e.g., images or video).
    - output: Folder to save the output results.
    - img_size: Size of the input image for detection.
    - conf_thres: Confidence threshold for detection.
    - iou_thres: Intersection over Union threshold for Non-Maximum Suppression.
    - device: Device for detection.
    """
    command = [
        'python', 'src/detect.py',  # Running the detection script
        '--cfg', 'cfg/yolov3-custom.cfg',  # Config for YOLOv3
        '--names', 'data/customdata/custom.names',  # Class names
        '--weights', 'weights/best.pt',  # Pre-trained weights
        '--source', 'data/images',  # Source directory for input images
        '--output', 'output/detections',  # Output directory for results
        '--img-size', '512',  # Image size for detection
        '--conf-thres', '0.3',  # Confidence threshold for detections
        '--iou-thres', '0.6',  # IOU threshold for NMS
        '--device', '0'  # GPU 0
    ]
    subprocess.run(command)  # Execute the detection command
    print(f"Detection completed. .")

if __name__ == '__main__':
    mode = sys.argv[1]  # Retrieve the mode argument (train or detect)

    if mode == "train":
        run_train()  # Run training
    elif mode == "detect":
        run_detect()  # Run detection

