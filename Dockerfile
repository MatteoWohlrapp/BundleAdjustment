# Use an official Ubuntu as a parent image
FROM ubuntu:latest

ENV DEBIAN_FRONTEND=noninteractive

# Set the working directory in the container
WORKDIR /usr/src/app

# Install any needed packages
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    libeigen3-dev \
    libgoogle-glog-dev \
    libgflags-dev \
    libceres-dev \
    libfreeimage-dev \
    libopencv-dev \
    python3-opencv \
    xvfb \
    x11vnc \
    libpcl-dev \
    clang \
    lldb \
    && rm -rf /var/lib/apt/lists/*

# Install additional dependencies for OpenCV and contrib modules
RUN apt-get update && apt-get install -y \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libgtk2.0-dev \
    libgtk-3-dev \
    libv4l-dev \
    libxvidcore-dev \
    libx264-dev \
    && rm -rf /var/lib/apt/lists/*

ENV DISPLAY=:99

# Clone OpenCV and OpenCV contrib
RUN git clone https://github.com/opencv/opencv.git \
    && git clone https://github.com/opencv/opencv_contrib.git

# Build and install OpenCV with contrib modules
RUN cd opencv \
    && mkdir build \
    && cd build \
    && cmake -D CMAKE_BUILD_TYPE=RELEASE \
    -D CMAKE_INSTALL_PREFIX=/usr/local \
    -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules \
    -D OPENCV_ENABLE_NONFREE=ON \
    -D BUILD_opencv_java=OFF \ 
    .. \
    && make -j$(nproc) \
    && make install \
    && ldconfig 

# Clone and build googletest
RUN git clone https://github.com/google/googletest.git --branch release-1.11.0 && \
    cd googletest && \
    cmake . && \
    make && \
    make install

# Copy the current directory contents into the container at /usr/src/app
COPY . .
