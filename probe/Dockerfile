FROM python:3.11-slim-bookworm
ARG SOURCE_REVISION=unknown
LABEL org.opencontainers.image.source="https://github.com/BigMy/wan21-vace-multitalk-probe"
LABEL org.opencontainers.image.revision=$SOURCE_REVISION
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 TOKENIZERS_PARALLELISM=false \
    HF_HUB_DISABLE_TELEMETRY=1 NVIDIA_VISIBLE_DEVICES=all NVIDIA_DRIVER_CAPABILITIES=compute,utility
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libgl1 libglib2.0-0 libgomp1 openssh-server rsync curl git ca-certificates \
    && rm -rf /var/lib/apt/lists/* && mkdir -p /run/sshd
RUN pip install --upgrade pip && pip install \
    torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 \
    --index-url https://download.pytorch.org/whl/cu128
COPY probe/requirements.txt /tmp/probe-requirements.txt
RUN pip install -r /tmp/probe-requirements.txt
WORKDIR /opt/Wan2GP
COPY . .
RUN python probe/import_check.py && pip freeze > /opt/environment-freeze.txt \
    && printf '%s\n' "$SOURCE_REVISION" > /opt/source-revision.txt
ENV PATH="/usr/local/bin:$PATH"
CMD ["sleep", "infinity"]
