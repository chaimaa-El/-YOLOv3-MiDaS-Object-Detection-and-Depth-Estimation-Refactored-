import configparser as cp
import subprocess
import sys
import os

# Load configuration settings
CONFIG_PATH = "src/cfg/mde.cfg"

def load_config(config_path):
    """
    Loads configuration settings from a given file.
    :param config_path: Path to the configuration file.
    :return: ConfigParser object with loaded settings.
    """
    config = cp.RawConfigParser()
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    config.read(config_path)
    return config

# Global configuration object
config = load_config(CONFIG_PATH)



# Training function
def run_train():
    """
    Trains the YOLOv3 model using the custom dataset as per configuration settings.
    """
    command = [
        'python', 'src/train.py',
        '--cfg', config.get('training', 'cfg', fallback='src/cfg/yolov3.cfg'),
        '--data', config.get('dataset', 'train_path', fallback='data/customdata/custom.data'),
        '--epochs', config.get('training', 'epochs', fallback='100'),
        '--batch-size', config.get('training', 'batch_size', fallback='16'),
        '--weights', config.get('training', 'weights', fallback='weights/best.pt'),
        '--device', config.get('settings', 'device', fallback='cpu')
    ]
    subprocess.run(command)
    print("Training completed.")



# Function to run object detection and depth estimation
def run_detect():
    """
    Runs object detection and depth estimation on input data as per configuration settings.
    """
    command = [
        'python', 'src/detect.py',
        '--cfg', config.get('yolo', 'cfg', fallback='cfg/yolov3-custom.cfg'),
        '--names', config.get('dataset', 'names', fallback='data/customdata/custom.names'),
        '--weights', config.get('output', 'weights', fallback='weights/best.pt'),
        '--source', config.get('input', 'source', fallback='data/images'),
        '--output', config.get('output', 'dir', fallback='output/detections'),
        '--img-size', config.get('settings', 'img_size', fallback='512'),
        '--conf-thres', config.get('settings', 'conf_thres', fallback='0.5'),
        '--iou-thres', config.get('settings', 'iou_thres', fallback='0.5'),
        '--device', config.get('settings', 'device', fallback='cpu')
    ]
    subprocess.run(command)
    print("Detection completed.")



# Main entry point
if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python main.py <mode>")
        print("Modes: train, detect")
        sys.exit(1)

    mode = sys.argv[1].lower()

    if mode == "train":
        run_train()
    elif mode == "detect":
        run_detect()
    else:
        print(f"Invalid mode: {mode}. Please use 'train' or 'detect'.")
        sys.exit(1)

