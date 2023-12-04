# syntax=docker/dockerfile:1
# Keep this syntax directive! It's used to enable Docker BuildKit


# ========================================================================================================= #
# BASE
# Sets up all our shared environment variables.
# ========================================================================================================= #
# Use official nvidia base image for pytorch https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch
#FROM nvcr.io/nvidia/pytorch:23.10-py3 as base
#FROM python:3.11.6 as python-base


FROM ubuntu:20.04 as base

#COPY --from=python-base /usr/local/ /usr/local/

# Python and Poetry Setup
ENV DEBIAN_FRONTEND=noninteractive \
    # ===== Python ===== #
    PYTHONUNBUFFERED=1 \

    # base path used by python to search for modules
    PROJECT_BASE_PATH="/workspaces/real-time-hand-gesture-detection" \
    # ===== Pip ===== #
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100 \

    # ===== Poetry ===== #
    # variables to control poetry's behavior, used by poetry automatically
    # https://python-poetry.org/docs/configuration/#using-environment-variables
    POETRY_VERSION=1.6.1 \
    # make poetry install to this location
    POETRY_HOME="/opt/poetry" \
    # do not ask any interactive question
    POETRY_NO_INTERACTION=1 \
    # never create virtual environment automaticly, only use env prepared by us
    POETRY_VIRTUALENVS_CREATE=false \
    # set up dependancy cache 
    POETRY_CACHE_DIR=/root/.cache \
    # venv to isolate poetry installation.
    POETRY_VIRTUALENVS_PATH="/venv" 

# prepend poetry and venv to path
ENV PATH="$POETRY_HOME/bin:$POETRY_VIRTUALENVS_PATH/bin:$PATH"

# append base python path to python path
ENV PYTHONPATH="$PROJECT_BASE_PATH:$PYTHONPATH"

RUN echo "Base image build successfully"


# ========================================================================================================= #
# BUILDER
# Used to build deps + create our virtual environment
# ========================================================================================================= #
FROM base as builder

# System dependencies
RUN apt-get update && apt-get install -y \
    software-properties-common \
    build-essential && \
    add-apt-repository ppa:deadsnakes/ppa 

RUN apt-get install -y \
    python3.10 \
    python3.10-distutils \
    python3.10-venv \
    git \
    sudo \
    # cleanup, remove unecessary packages
    && rm -rf /var/lib/apt/lists/*

# If USE_CUDA is true, include CUDA packages
# ARG USE_GPU=false

#nvidia-cuda-dev-11.8.0 \
#nvidia-cuda-toolkit-11.8.0 \
# ARG CUDA_VERSION=11.8.0

# Download and install CUDA
# RUN if [ "$USE_GPU" = "true" ]; then \
#     apt-get update && apt-get install -y wget && \
#     wget https://developer.download.nvidia.com/compute/cuda/${CUDA_VERSION}/local_installers/cuda_${CUDA_VERSION}_520.61.05_linux.run && \
#     chmod +x cuda_${CUDA_VERSION}_520.61.05_linux.run && \
#     ./cuda_${CUDA_VERSION}_520.61.05_linux.run --toolkit --silent --override && \
#     rm cuda_${CUDA_VERSION}_520.61.05_linux.run && \
#     echo "Using NVIDIA GPU"; \
# fi

# Create virtual environment
RUN python3.10 -m venv $POETRY_VIRTUALENVS_PATH

# Upgrade pip and setuptools
RUN $POETRY_VIRTUALENVS_PATH/bin/pip install -U pip setuptools

# Install poetry - respects $POETRY_VERSION & $POETRY_HOME
RUN $POETRY_VIRTUALENVS_PATH/bin/pip install poetry

# used to init dependencies
COPY poetry.lock pyproject.toml ./

# Install only runtime deps to POETRY_VIRTUALENVS_PATH.
# The --mount will mount the buildx cache directory to where
# Poetry and Pip store their cache so that they can re-use it.
RUN --mount=type=cache,target=$POETRY_CACHE_DIR \
    $POETRY_VIRTUALENVS_PATH/bin/poetry install --no-root --only main

RUN echo "Builder image build successfully"


# ========================================================================================================= #
# DEVELOPMENT
# Image used during development / testing.
# Devcontainer automatically mounts local root of the project with .git into container.
# ========================================================================================================= #
FROM builder as development

# copy virtual env width runtime deps from builder
COPY --from=builder ${POETRY_VIRTUALENVS_PATH} ${POETRY_VIRTUALENVS_PATH}

# Quick install of additional dev deps as runtime deps come cached from builder already.
RUN poetry install --no-root

RUN echo "Development image build successfully"

