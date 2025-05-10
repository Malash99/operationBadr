FROM osrf/ros:noetic-desktop-full
ENV PYTHONUNBUFFERED 1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-opencv \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY docker/requirements.txt /tmp/
RUN pip3 install --upgrade pip && \
    pip3 install -r /tmp/requirements.txt

# Initialize ROS workspace
RUN mkdir -p /catkin_ws/src
WORKDIR /catkin_ws
RUN /bin/bash -c "source /opt/ros/noetic/setup.bash && catkin_make"

WORKDIR /app
COPY . /app