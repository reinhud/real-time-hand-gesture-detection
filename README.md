# Real Time Hand Gesture Detection

This project is a real time hand gesture detection using Pytorch. The project is based on the following paper: https://arxiv.org/abs/1901.10323

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

#### Dependencies
Please make sure you have the following dependencies set up on your system:
1. [VSCode](https://code.visualstudio.com/download)
2. [DevContainer Extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers)
3. [Docker](https://docs.docker.com/docker-for-windows/install/)
4. [CUDA Toolkit](https://docs.nvidia.com/cuda/wsl-user-guide/index.html)
5. [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
###### Windows only
6. [WSL Extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-wsl)
7. [WSL2](https://learn.microsoft.com/en-us/windows/wsl/install)

#### How to set up dev environment
1. Clone the repository.
2. Open the repository in VSCode.
    - if you are using windows, pls open the project using wsl by either starting vscode
    from within wsl or use the VSCode WSL extension to open the project in WSL
3. Launch project in Devcontainer.
    - Open VsCode command palette(strg+p)
    - run ```Dev Containers: Reopen in Container```

## 4. Usage
To interact with the project, you can use the following commands for common usecases:

```bash
# Show help
make help

# Clean the project
make clean

# Run the tests
make test

# Run the tests with coverage
make coverage

# Run the linter
make lint

# Run the formatter
make format

# Add a package to the project
poetry add <package>

# Lock the dependencies
poetry lock

```


## 5. Licence
This project is licensed under the MIT licence - see LICENSE file for details.


## 6. Acknowledgments