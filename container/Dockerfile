ARG DISTRIBUTION_ROS=humble
ARG VERSION_PYTHON=3.10



# ========================================
FROM amd64/ros:"$DISTRIBUTION_ROS-perception" AS base-amd64



# ========================================
FROM arm64v8/ros:"$DISTRIBUTION_ROS-perception" AS base-arm64



# ========================================
FROM "base-$TARGETARCH" AS system1

# Set shell
SHELL ["/bin/bash", "-c"]

# Configure shell
ARG DEBIAN_FRONTEND=noninteractive
ENV DEBCONF_NOWARNINGS="yes"

# Install essentials
RUN apt-get update -qq && \
    apt-get install -y -qq --no-install-recommends \
    apt-utils \
    build-essential \
    ca-certificates \
    curl \
    git \
    gnupg \
    lsb-release \
    nano \
    software-properties-common \
    tmux \
    unzip \
    wget \
    x11-apps \
    && \
    apt-get autoremove -y -qq && \
    apt-get clean -qq && \
    rm -rf /var/lib/apt/lists/*

# Enable GPU
ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=all

# Install CUDA Toolkit
RUN wget --quiet "https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb" --directory-prefix /tmp/ && \
    dpkg -i /tmp/cuda-keyring_1.1-1_all.deb && \
    rm -rf /tmp/*
RUN apt-get update -qq && \
    apt-get install -y -qq --no-install-recommends \
    cuda-toolkit-12-4 \
    && \
    apt-get autoremove -y -qq && \
    apt-get clean -qq && \
    rm -rf /var/lib/apt/lists/*
ENV PATH="/usr/local/cuda/bin:$PATH"
RUN ldconfig

# Install cuDNN
RUN apt-get update -qq && \
    apt-get install -y -qq --no-install-recommends \
    cudnn9-cuda-12  \
    && \
    apt-get autoremove -y -qq && \
    apt-get clean -qq && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies
ARG DISTRIBUTION_ROS=humble
RUN apt-get update -qq && \
    apt-get install -y -qq --no-install-recommends \
    gettext-base \
    ros-dev-tools \
    "ros-$DISTRIBUTION_ROS-moveit-msgs" \
    "ros-$DISTRIBUTION_ROS-rmw-cyclonedds-cpp" \
    "ros-$DISTRIBUTION_ROS-rosbag2" \
    "ros-$DISTRIBUTION_ROS-vision-msgs" \
    && \
    apt-get autoremove -y -qq && \
    apt-get clean -qq && \
    rm -rf /var/lib/apt/lists/*

# Add group and user
ARG USER=user \
    UID=1001
RUN addgroup --gid "$UID" "$USER" && \
    adduser --uid "$UID" --gid "$UID" --disabled-password "$USER" && \
    usermod -aG sudo "$USER" && \
    echo "$USER ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers



# ========================================
FROM system1 AS system-amd64

# Install TensorRT
RUN apt-get update -qq && \
    apt-get install -y -qq --no-install-recommends \
    tensorrt \
    && \
    apt-get autoremove -y -qq && \
    apt-get clean -qq && \
    rm -rf /var/lib/apt/lists/*

# Install torch2trt plugins
RUN git clone https://github.com/NVIDIA-AI-IOT/torch2trt /opt/torch2trt
RUN cd /opt/torch2trt && \
    cmake -B build . && \
    cmake --build build --target install && \
    ldconfig



# ========================================
FROM system1 AS system-arm64

# Install TensorRT
# Assume Jetson device
# Full installer only available for amd64
RUN wget --quiet https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/10.0.1/tars/TensorRT-10.0.1.6.l4t.aarch64-gnu.cuda-12.4.tar.gz --output-document /tmp/TensorRT.tar
RUN tar -xvf /tmp/TensorRT.tar --directory /usr/src && \
    mv /usr/src/TensorRT-* /usr/src/tensorrt && \
    cp -r /usr/src/tensorrt/lib/* "/usr/lib/$(uname -m)-linux-gnu/" && \
    cp -r /usr/src/tensorrt/include/* "/usr/include/$(uname -m)-linux-gnu/" && \
    rm -rf /tmp/*

# Install torch2trt plugins
RUN git clone https://github.com/NVIDIA-AI-IOT/torch2trt /opt/torch2trt
RUN CUDA_ARCHITECTURES=87 && \
    sed 's|^set(CUDA_ARCHITECTURES.*|#|g' -i /opt/torch2trt/CMakeLists.txt && \
    sed 's|Catch2_FOUND|False|g' -i /opt/torch2trt/CMakeLists.txt && \
    cmake -B /opt/torch2trt/build /opt/torch2trt && \
    cmake --build /opt/torch2trt/build --target install && \
    ldconfig



# ========================================
FROM "system-$TARGETARCH" AS python

# Install Python
RUN apt-get update -qq && \
    apt-get install -y -qq --no-install-recommends \
    python-is-python3 \
    python3 \
    python3-pip \
    python3-venv \
    && \
    apt-get autoremove -y -qq && \
    apt-get clean -qq && \
    rm -rf /var/lib/apt/lists/*

# Install Python version explicitly
ARG VERSION_PYTHON=3.10
RUN add-apt-repository -y ppa:deadsnakes/ppa
RUN apt-get update -qq && \
    apt-get install -y -qq --no-install-recommends \
    "python$VERSION_PYTHON" \
    "python$VERSION_PYTHON-venv" \
    && \
    apt-get autoremove -y -qq && \
    apt-get clean -qq && \
    rm -rf /var/lib/apt/lists/*

# Configure Python
ENV PYTHONUNBUFFERED=1 \
    PYTHONWARNINGS=ignore:::setuptools.command.install,ignore:::setuptools.command.easy_install,ignore:::setuptools.command.develop,ignore:::setuptools.dist,ignore:::pkg_resources \
    PIP_ROOT_USER_ACTION=ignore \
    PIP_DISABLE_PIP_VERSION_CHECK=on

# Setup venv
RUN "python$VERSION_PYTHON" -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONPATH="/opt/venv/lib/python$VERSION_PYTHON/site-packages:$PYTHONPATH"
RUN pip install --no-cache-dir \
    pip==24.2 \
    setuptools==74.0.0 \
    wheel==0.44.0



# ========================================
FROM python AS libs-amd64

# Install dependencies
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --requirement /tmp/requirements.txt && \
    rm -rf /tmp/*



# ========================================
FROM libs AS libs1-arm64

# Install dependencies
COPY requirements.txt /tmp/requirements.txt
RUN grep -ivE "accelerate==|tensorrt==|torch==|torch-tensorrt==|torchaudio==|torchprofile==|torchvision==" /tmp/requirements.txt > /tmp/requirements_arm.txt
RUN pip install --no-cache-dir --requirement /tmp/requirements_arm.txt && \
    rm -rf /tmp/*

# Install dependencies for pytorch wheel
RUN apt-get update -qq && \
    apt-get install -y -qq --no-install-recommends \
    libopenblas-dev \
    libopenmpi-dev \
    libomp-dev \
    && \
    apt-get autoremove -y -qq && \
    apt-get clean -qq && \
    rm -rf /var/lib/apt/lists/*

# Install PyTorch, TorchVision, TorchAudio
# Assume Jetson device
# Check updates here: https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048
RUN wget --quiet https://nvidia.box.com/shared/static/zvultzsmd4iuheykxy17s4l2n91ylpl8.whl --output-document /tmp/torch-2.3.0-cp310-cp310-linux_aarch64.whl && \
    wget --quiet https://nvidia.box.com/shared/static/u0ziu01c0kyji4zz3gxam79181nebylf.whl --output-document /tmp/torchvision-0.18.0a0+6043bc2-cp310-cp310-linux_aarch64.whl && \
    wget --quiet https://nvidia.box.com/shared/static/9si945yrzesspmg9up4ys380lqxjylc3.whl --output-document /tmp/torchaudio-2.3.0+952ea74-cp310-cp310-linux_aarch64.whl
RUN pip install --no-cache-dir \
    /tmp/torch-2.3.0-cp310-cp310-linux_aarch64.whl \
    /tmp/torchvision-0.18.0a0+6043bc2-cp310-cp310-linux_aarch64.whl \
    /tmp/torchaudio-2.3.0+952ea74-cp310-cp310-linux_aarch64.whl \
    && \
    rm -rf /tmp/*

# Install TensorRT
RUN pip install --no-cache-dir \
    /usr/src/tensorrt/python/tensorrt-*-cp310-*.whl



# ========================================
FROM libs1-arm64 AS libs2-arm64-torch_tensorrt-build

# Install Bazel
RUN BAZELISK_RELEASE=$(wget -qO- https://api.github.com/repos/bazelbuild/bazelisk/releases/latest | grep -Po '"tag_name": "\K.*?(?=")') && \
    BAZELISK_URL="https://github.com/bazelbuild/bazelisk/releases/download/$BAZELISK_RELEASE/bazelisk-linux-arm64" && \
    echo "BAZELISK_RELEASE=$BAZELISK_RELEASE" && echo "BAZELISK_URL=$BAZELISK_URL" && \
    wget --quiet --show-progress --progress=bar:force:noscroll --no-check-certificate $BAZELISK_URL -O /usr/local/bin/bazel && \
    chmod +x /usr/local/bin/bazel

# Build Torch-TensorRT
RUN git clone --branch=v2.3.0 https://github.com/pytorch/TensorRT /opt/torch_tensorrt
WORKDIR /opt/torch_tensorrt
COPY resources/WORKSPACE.jp60 /opt/torch_tensorrt/toolchains/jp_workspaces/WORKSPACE.jp60
COPY resources/torch_tensorrt.diff /opt/torch_tensorrt/
RUN git apply torch_tensorrt.diff
RUN python setup.py bdist_wheel --jetpack-version 6.0 --use-cxx11-abi
RUN cp dist/torch_tensorrt*.whl /opt/



# ========================================
FROM libs1-arm64 AS libs-arm64

# Install Torch-TensorRT
COPY --from=libs2-arm64-torch_tensorrt-build /opt/torch_tensorrt*.whl /opt/
RUN pip install --no-cache-dir \
    /opt/torch_tensorrt*.whl \
    nvidia-modelopt[all]~=0.17.0 --extra-index-url https://pypi.nvidia.com



# ========================================
FROM "libs-$TARGETARCH" AS monocular_depth

# Add code
ARG USER=user \
    UID=1001
RUN git clone https://github.com/bertan-karacora/ros2_utils.git "/home/$USER/colcon_ws/src/ros2_utils"
COPY . "/home/$USER/colcon_ws/src/ros2_monocular_depth"
RUN chown --recursive "$USER:$USER" "/home/$USER"

# Build
ARG DISTRIBUTION_ROS=humble
USER "$USER"
WORKDIR "/home/$USER/colcon_ws"
RUN source "/opt/ros/$DISTRIBUTION_ROS/setup.bash" && \
    colcon build --symlink-install

# Setup run
WORKDIR "/home/$USER/colcon_ws/src/ros2_monocular_depth"
ENTRYPOINT ["container/entrypoint.sh"]
CMD ["scripts/start_all.sh"]
