# CPU image by default. For CUDA: --build-arg BASE_IMAGE=nvidia/cuda:12.8.1-runtime-ubuntu24.04 --build-arg TORCH_VARIANT=cu128
ARG BASE_IMAGE=python:3.12-slim
FROM ${BASE_IMAGE}
ARG TORCH_VARIANT=cpu

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl unzip \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.9.5 /uv /uvx /usr/local/bin/
ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PYTHON=3.12 \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH="/opt/venv/bin:${PATH}"

WORKDIR /workspace
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --extra ${TORCH_VARIANT} --no-install-project
COPY . .
RUN uv sync --frozen --extra ${TORCH_VARIANT}

CMD ["bash"]
