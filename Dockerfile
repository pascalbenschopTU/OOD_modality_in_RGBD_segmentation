# Use an official Anaconda image as a base
FROM continuumio/miniconda3

# Set the working directory inside the container
WORKDIR /app

# Copy the environment.yml and requirements.txt to the container
COPY environment_mac.yml /app/environment.yml
COPY requirements_docker.txt /app/requirements.txt

RUN apt-get update && apt-get install -y \
    g++ \
    gcc \
    cmake \
    libjpeg-dev \
    zlib1g-dev \
    libgl1-mesa-glx

# Create a conda environment from the environment.yml file
RUN conda env create -f /app/environment.yml

# Activate the conda environment and ensure it's activated in subsequent steps
SHELL ["conda", "run", "-n", "depth", "/bin/bash", "-c"]

RUN python -m pip install --upgrade pip

# Install additional packages via pip (if needed)
RUN python -m pip install --no-deps -r /app/requirements.txt --root-user-action=ignore

# Copy the rest of your project files into the container
COPY . /app

# Make sure the Python script is executable
RUN chmod +x /app/utils/train_day_night.py

# Set the default command to run your Python script using conda
CMD ["conda", "run", "-n", "depth", "python", "/app/utils/train_day_night.py"]
