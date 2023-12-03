# syntax=docker/dockerfile:1
# Keep this syntax directive! It's used to enable Docker BuildKit


# ========================================================================================================= #
# BASE
# Sets up all our shared environment variables.
# ========================================================================================================= #
# Use official nvidia base image for pytorch https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch
FROM nvcr.io/nvidia/pytorch:23.10-py3 as base

# Python and Poetry Setup
ENV PYTHONUNBUFFERED=1 \
    # pip
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100 \
    # Poetry
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
    POETRY_VIRTUALENVS_PATH="/venv" \
    # base path used by python to search for modules
    PROJECT_BASE_PATH="/workspaces/real-time-hand-gesture-detection"

# prepend poetry, venv and base python path to path
ENV PATH="$POETRY_HOME/bin:$POETRY_VIRTUALENVS_PATH/bin:$PROJECT_BASE_PATH:$PATH"

RUN echo "Base image build successfully"


# ========================================================================================================= #
# BUILDER
# Used to build deps + create our virtual environment
# ========================================================================================================= #
FROM base as builder

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

# Create virtual environment
RUN python3 -m venv $POETRY_VIRTUALENVS_PATH

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
    $POETRY_VIRTUALENVS_PATH/bin/poetry install --no-root --no-dev

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


# ========================================================================================================= #
# PRODUCTION
# ========================================================================================================= #
FROM builder as production

# copy virtual env width deps from builder
COPY --from=builder ${POETRY_VIRTUALENVS_PATH} ${POETRY_VIRTUALENVS_PATH}

# copy project
COPY . /workspaces/real-time-hand-gesture-detection/

RUN echo "Production image build successfully"