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
├── src/              # contains all the core modules and scripts for the project
│   ├── cfg/          # contains configuration files
│   ├── utils/        # contains utility and helper functions
│   ├── model/        # contains the model definition files (for YOLOv3 and MiDaS)
│   ├── train.py      # Script for training the  the MDENet (Monocular Depth Estimation Network) model with custom dataset   
│   ├── detect.py     # Script for running inference and object detection with depth estimation   
│
├── notebooks/        # Folder containing Jupyter notebooks for experimentation  
│
├── Input/            # Dataset folder ( include images, annotations, etc.)
│
├── output/           # Folder to store output results from the detection script
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
git clone https://github.com/chaimaa-El/YOLOv3-MiDaS-Object-Detection-and-Depth-Estimation-Refactored.git
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
### main.py

The main.py script is the entry point for running the project. It supports two main modes: train and detect.

    --mode train: This mode starts the training process for the YOLOv3 model.
    --mode detect: This mode runs object detection and depth estimation on input images.

### train.py

The train.py script is designed to train the MDENet (Monocular Depth Estimation Network) model using a custom dataset for monocular depth prediction. The training process incorporates various advanced techniques such as mixed precision training, multi-scale augmentation, and learning rate scheduling. This script also supports distributed training across multiple GPUs using PyTorch's distributed data parallelism.

### detect.py

The detect.py script performs object detection and depth estimation using a custom-trained YOLO-based model. It leverages a multi-stage pipeline that integrates YOLO for object detection and an additional depth estimation model (MDENet). The script is configured through a .cfg file and performs inference on images or videos. It outputs detection results and depth maps for each frame processed.

### utils.py

The utils.py file contains utility functions and imports related to computer vision tasks, specifically for object detection using PyTorch and other libraries. It is heavily used in YOLOv5-like models and other deep learning tasks that involve image processing, model evaluation, and loss computation.


