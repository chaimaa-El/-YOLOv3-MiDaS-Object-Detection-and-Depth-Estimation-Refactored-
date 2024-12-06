# YOLOv3 + MiDaS for Object Detection and Depth Estimation

## Overview

This repository contains a refactored and organized version of the original sarvan0506/yolo-midas project for object detection and depth estimation. The original project was a collection of Jupyter notebooks without proper structure. This version organizes the code into a clean and modular format and with more maintainable structure with clearer documentation, making it easier for users to experiment with and deploy object detection and depth estimation for real-world applications, such as autonomous navigation and robotic.

This project combines **YOLOv3** (a state-of-the-art object detection algorithm) with **MiDaS** (a deep learning model for monocular depth estimation) to perform **joint object detection and depth estimation**. The goal is to detect objects in images and estimate their **depth** (distance from the camera), which is crucial for applications such as **autonomous navigation**, **robotics**, and **self-driving cars**.

The integration of these two models provides the ability to detect and measure the distance of objects in real-time from input images.

---

## Project Structure

The repository is organized as follows:

```bash
yolo-midas/
│
├── src/              # Core source code for training and detection
│   ├── train.py      # Script for training the YOLOv3 model with custom dataset
│   ├── detect.py     # Script for running inference on images or videos
│   ├── utils.py      # Utility functions for data preprocessing and postprocessing
│
├── notebooks/        # Folder containing Jupyter notebooks for experimentation
│   ├── exploration.ipynb   # Jupyter notebook for testing and visualizing model outputs
│
├── cfg/              # Folder for configuration files
│   ├── yolov3-custom.cfg    # YOLOv3 model config file
│   ├── mde.cfg       # MiDaS depth estimation model config
│
├── data/             # Dataset folder (can include images, annotations, etc.)
│   ├── customdata/   # Custom data folder (images, labels, etc.)
│   ├── images/       # Folder containing images for detection
│
├── weights/          # Folder for storing model weights
│  
│
├── output/           # Folder to store output results from the detection script
│   ├── detections/   # Folder to store the detection results
│
├── requirements.txt  # List of dependencies needed to run the project
├── main.py           # Main entry point for running the project (train or detect)
├── README.md         # Project documentation file

```

---

## Installation and Setup

### 1. **Clone the Repository**
First, clone the repository to your local machine:
```bash
git clone https://github.com/sarvan0506/yolo-midas.git
cd yolo-midas
```

---

## Install Dependencies
Make sure you have Python 3.6+ installed. Then, use the requirements.txt file to install the necessary dependencies:

```bash
pip install -r requirements.txt
```
This will install all the required libraries for training, detection, and depth estimation.

---

## Running the Code
### 1. **Train the Model**

To train the YOLOv3 model on a custom dataset, run the following command from the terminal:

```bash
python main.py --mode train 
```
This will start the training process using the specified configuration .
### 2. **Run Object Detection and Depth Estimation**

To run inference (detection and depth estimation) on images, use the following command:


```bash
python main.py --mode detect 
```

This command will detect objects in images from the data/images folder, estimate their depth, and save the results in the output/detections folder.


---

## Main Functionalities

--YOLOv3 Object Detection: YOLOv3 detects objects in images based on a custom-trained model. It outputs the bounding boxes and labels of the detected objects.

--MiDaS Depth Estimation: MiDaS estimates the depth (distance from the camera) of each detected object, providing additional information for 3D understanding.

--Combined Output: The system generates output containing both object detection results and depth estimation information, which can be visualized or used for further processing.

---

## Optional Steps
GPU Usage

If you have a GPU and want to use it for training or detection, ensure you have CUDA installed. You can specify the device as cuda or the appropriate GPU index, for example:

python main.py --mode train --device cuda  # Use GPU for training
python main.py --mode detect --device cuda  # Use GPU for detection

If no GPU is available, the code will default to using the CPU.

---

## Code Description
main.py

The main.py script is the entry point for running the project. It supports two main modes: train and detect.

    --mode train: This mode starts the training process for the YOLOv3 model.
    --mode detect: This mode runs object detection and depth estimation on input images.

train.py

The train.py script is responsible for training the YOLOv3 model using a custom dataset. It takes in configuration settings, dataset details, and training parameters.
detect.py

The detect.py script performs object detection and depth estimation. It detects objects in input images or videos and calculates the depth of the objects using the MiDaS model.
utils.py

The utils.py file contains utility functions for pre-processing and post-processing data, such as image resizing, loading datasets, or processing results


