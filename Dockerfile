# syntax=docker/dockerfile:1
# Keep this syntax directive! It's used to enable Docker BuildKit


# ========================================================================================================= #
# BASE
# Sets up all our shared environment variables
# ========================================================================================================= #
# Use official nvidia base image for pytorch https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch
FROM nvcr.io/nvidia/pytorch:23.10-py3 as base

# Python and Poetry Setup
ENV PYTHONUNBUFFERED=1 \
    # pip
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100 \
    \
    # Poetry
    # https://python-poetry.org/docs/configuration/#using-environment-variables
    POETRY_VERSION=1.6.1 \
    # make poetry install to this location
    POETRY_HOME="/opt/poetry" \
    # do not ask any interactive question
    POETRY_NO_INTERACTION=1 \
    # never create virtual environment automaticly, only use env prepared by us
    POETRY_VIRTUALENVS_CREATE=false \
    \
    # this is where our requirements + virtual environment will live
    VIRTUAL_ENV="/venv" 

    # prepend poetry and venv to path
    ENV PATH="$POETRY_HOME/bin:$VIRTUAL_ENV/bin:$PATH"

    RUN echo "Base image build successfully"


# ========================================================================================================= #
# BUILDER-BASE
# Used to build deps + create our virtual environment
# ========================================================================================================= #
FROM base as builder-base

# Install dependencies
RUN apt-get update && \
    apt-get install -y \
    # system deps
    apt-transport-https \
    gnupg \
    ca-certificates \
    build-essential \
    git \
    curl \
    # python deps
    python3.10-venv && \
    # cleanup, remove unecessary packages
    rm -rf /var/lib/apt/lists/*

# prepare virtual env
RUN python -m venv $VIRTUAL_ENV

# install poetry - respects $POETRY_VERSION & $POETRY_HOME
# The --mount will mount the buildx cache directory to where
# Poetry and Pip store their cache so that they can re-use it
RUN --mount=type=cache,target=/root/.cache \
    curl -sSL https://install.python-poetry.org | python -

# used to init dependencies
COPY poetry.lock pyproject.toml ./

# install runtime deps to VIRTUAL_ENV
RUN --mount=type=cache,target=/root/.cache \
    poetry install --no-root --only main

RUN echo "Builder base image build successfully"


# ========================================================================================================= #
# DEVELOPMENT
# Image used during development / testing
# ========================================================================================================= #
FROM builder-base as development

# install dev deps to VIRTUAL_ENV
RUN --mount=type=cache,target=/root/.cache \
    poetry install --no-root

# copy in our source code last, as it changes the most
COPY . .

RUN echo "Development image build successfully"


# ========================================================================================================= #
# Production
# ========================================================================================================= #
