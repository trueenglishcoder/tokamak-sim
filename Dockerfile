FROM python:3.11-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MPLCONFIGDIR=/tmp/matplotlib

WORKDIR /app

ARG INSTALL_GPU=false

RUN apt-get -o Acquire::Retries=5 update \
    && apt-get install -y --no-install-recommends -o Acquire::Retries=5 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY tokamak_control ./tokamak_control
COPY scripts ./scripts

RUN python -m pip install --upgrade pip \
    && python -m pip install . \
    && if [ "$INSTALL_GPU" = "true" ]; then python -m pip install ".[gpu]"; fi

COPY docs ./docs

RUN mkdir -p /app/configs /app/data /app/runs /app/output /tmp/matplotlib

CMD ["python", "scripts/run_simulation_artifacts.py", "--help"]
