# Real Time Hand Gesture Detection

This project is a real time hand gesture detection using Pytorch.

## Contents
- [1. Introduction and Overview](#1-introduction-and-overview)
- [2. Key Features](#2-getting-started)
- [3. Getting Started](#2-getting-started)
- [4. Usage](#4-usage)
- [5. Licence](#3-licence)
- [6. Acknowledgments](#4-acknowledgments)


## 1. Introduction and Overview
Welcome to the Real-Time Hand Gesture Detection project! This project is part of a university assignment aimed at implementing a real-time hand gesture detection system.


## 2. Key Features
- Detection of hand gestures from a video source
- Distinction between gestures and random movements
- Detection and classification should happen in real time


## 3. Getting Started
To get started with the project, follow the instructions below.

``` shell
# create and activate the conda environment
conda create -n gesture python=3.10
conda activate gesture

# install poetry and install dependencies
conda install poetry
cd /path/to/this/repo
poetry install
```

## 4. Usage

To interact with the project, you can use the following commands for common usecases:

### Training

``` shell
poetry run python gesture_detection.cli fit --config gesture_detection/config/model_config/lstm.py
```

### Inference

This currently only works with LSTM checkpoints. 
A few checkpoints from the LSTM trained on Jester Gesture is available in the release section. 
This model is not expected to generalize well on out-of-distribution data, i.e., your camera. 
However, you can still try this out and see if you're lucky ;)

``` shell
poetry run python gesture_detection.inference --ckpt /path/to/lstm.ckpt
```

## 5. Licence
This project is licensed under the MIT licence - see LICENSE file for details.


## 6. Acknowledgments

This project would not have been possible without the creators of the IPNHand, NVGesture and Jester Gesture dataset. 