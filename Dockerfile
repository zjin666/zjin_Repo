ARG DOCKER_PLATFORM=linux/amd64
FROM --platform=${DOCKER_PLATFORM} nvcr.io/nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

ENV TZ=Asia/Shanghai \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple \
    PIP_TRUSTED_HOST=mirrors.aliyun.com \
    PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cu121 \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/local/cuda/lib64:/usr/local/cuda-12.0/targets/x86_64-linux/lib:/usr/lib/x86_64-linux-gnu:/usr/local/cuda/compat

RUN mkdir -p /app /saisresult /app/models

RUN set -eux; \
    if [ -f /etc/apt/sources.list ]; then \
        sed -i \
            -e 's|http://archive.ubuntu.com/ubuntu/|https://mirrors.aliyun.com/ubuntu/|g' \
            -e 's|http://security.ubuntu.com/ubuntu/|https://mirrors.aliyun.com/ubuntu/|g' \
            -e 's|https://archive.ubuntu.com/ubuntu/|https://mirrors.aliyun.com/ubuntu/|g' \
            -e 's|https://security.ubuntu.com/ubuntu/|https://mirrors.aliyun.com/ubuntu/|g' \
            /etc/apt/sources.list; \
    fi; \
    find /etc/apt/sources.list.d -type f \( -name '*.list' -o -name '*.sources' \) -exec sed -i \
        -e 's|http://archive.ubuntu.com/ubuntu/|https://mirrors.aliyun.com/ubuntu/|g' \
        -e 's|http://security.ubuntu.com/ubuntu/|https://mirrors.aliyun.com/ubuntu/|g' \
        -e 's|https://archive.ubuntu.com/ubuntu/|https://mirrors.aliyun.com/ubuntu/|g' \
        -e 's|https://security.ubuntu.com/ubuntu/|https://mirrors.aliyun.com/ubuntu/|g' \
        {} +; \
    apt-get update; \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        python3-venv \
        tini \
        bash \
        wget \
        ca-certificates \
        libomp-dev \
        libgomp1 \
        libglib2.0-0 \
        libgl1 \
        libsm6 \
        libxrender1 \
        libxext6; \
    rm -rf /var/lib/apt/lists/*

RUN set -eux; \
    mkdir -p /usr/local/cuda/lib64; \
    echo "/usr/local/cuda/lib64" > /etc/ld.so.conf.d/cuda.conf; \
    for lib in libcublas libcublasLt libcudnn; do \
        target="$(find -H /usr/local/cuda /usr/local/cuda-* /usr/lib -name "${lib}.so.*" 2>/dev/null | sort -V | tail -n 1 || true)"; \
        if [ -n "$target" ]; then \
            ln -sf "$target" "/usr/local/cuda/lib64/${lib}.so"; \
            echo "Linked /usr/local/cuda/lib64/${lib}.so -> $target"; \
        else \
            echo "Missing ${lib}.so.*"; \
        fi; \
    done; \
    ldconfig; \
    python3 -c "import ctypes; [ctypes.CDLL(x) for x in ('libcublas.so', 'libcublasLt.so', 'libcudnn.so')]; print('CUDA libraries load OK')"

WORKDIR /app

COPY requirements.txt /app/requirements.txt

ENV PIP_DEFAULT_TIMEOUT=180 \
    PIP_RETRIES=10 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_ROOT_USER_ACTION=ignore

RUN set -eux; \
    python3 -m pip install --upgrade "pip<25" setuptools wheel

# Install PyTorch with CUDA 12.1 support
RUN set -eux; \
    python3 -m pip install --prefer-binary \
        torch torchvision \
        --index-url https://download.pytorch.org/whl/cu121

# Install remaining dependencies
RUN set -eux; \
    python3 -m pip install --prefer-binary -r /app/requirements.txt

# Verify PyTorch + CUDA
RUN set -eux; \
    python3 -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda)"

COPY models/ /app/models/
COPY src/ /app/src/
COPY run.sh /app/run.sh
RUN chmod +x /app/run.sh

ENTRYPOINT ["/usr/bin/tini", "--", "bash", "/app/run.sh"]
